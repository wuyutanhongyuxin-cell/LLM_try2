from __future__ import annotations
"""Lighter DEX 实盘执行器 — 核心下单与仓位管理。"""
import asyncio
import os
import time
from decimal import Decimal

import lighter
from lighter import ApiClient, Configuration, SignerClient
from loguru import logger

from src.execution.lighter_helpers import (
    fetch_last_price, place_ioc_order, query_balance, query_position, wait_for_fill,
)
from src.execution.signal import Action, TradeSignal

BASE_URL = "https://mainnet.zklighter.elliot.ai"
SEND_ERROR_GRACE_TIMEOUT = 1.5


class LighterExecutor:
    """Lighter DEX 实盘执行器，管理下单和仓位追踪。"""

    def __init__(
        self, private_key: str = "", account_index: int = 0,
        api_key_index: int = 0, max_position: Decimal = Decimal("0.01"),
        fill_timeout: float = 3.0, min_balance: Decimal = Decimal("10"),
        dry_run: bool = False,
    ) -> None:
        self._private_key = private_key or os.environ.get("LIGHTER_API_KEY_PRIVATE_KEY", "")
        self._account_index, self._api_key_index = account_index, api_key_index
        self._max_position, self._fill_timeout = max_position, fill_timeout
        self._min_balance, self._dry_run = min_balance, dry_run
        self._signer: SignerClient | None = None
        self._api_client: ApiClient | None = None
        self._market_index: int | None = None
        self._base_mult = self._price_mult = 1
        self._ticker, self._last_price = "", Decimal("0")
        self._pending_fills: dict[int, asyncio.Event] = {}
        self._fill_results: dict[int, dict] = {}
        self._local_position = self._realized_pnl = Decimal("0")
        self._trade_count = self._consecutive_losses = 0

    async def connect(self, ticker: str) -> None:
        self._ticker = ticker
        self._api_client = ApiClient(configuration=Configuration(host=BASE_URL))
        self._signer = SignerClient(
            url=BASE_URL, account_index=self._account_index,
            api_private_keys={self._api_key_index: self._private_key},
        )
        err = self._signer.check_client()
        if err is not None:
            raise RuntimeError(f"Lighter SignerClient 检查失败: {err}")
        await self._fetch_market_config()
        self._local_position = await self.get_position()
        logger.info(f"Lighter 执行器就绪: {ticker} 仓位={self._local_position}")

    async def disconnect(self) -> None:
        if self._api_client:
            await self._api_client.close()

    async def _fetch_market_config(self) -> None:
        order_api = lighter.OrderApi(self._api_client)
        order_books = await order_api.order_books()
        for market in order_books.order_books:
            if market.symbol == self._ticker:
                self._market_index = market.market_id
                self._base_mult = pow(10, market.supported_size_decimals)
                self._price_mult = pow(10, market.supported_price_decimals)
                return
        raise RuntimeError(f"Lighter 找不到 ticker: {self._ticker}")

    async def execute_signal(self, signal: TradeSignal, current_price: Decimal) -> bool:
        """执行交易信号，返回是否成功。"""
        if signal.action == Action.HOLD:
            return False
        self._last_price = current_price
        balance = await self.get_balance()
        if balance < self._min_balance:
            logger.warning(f"余额不足: {balance} < {self._min_balance}")
            return False
        if signal.action == Action.BUY:
            return await self._execute_buy(signal, balance, current_price)
        if signal.action == Action.SELL:
            return await self._execute_sell(signal, current_price)
        return False

    async def _execute_buy(
        self, signal: TradeSignal, balance: Decimal, price: Decimal,
    ) -> bool:
        size_usd = balance * Decimal(str(signal.size_pct)) / Decimal("100")
        size_btc = size_usd / price
        remaining = self._max_position - self._local_position
        if remaining <= 0:
            logger.warning("已达最大仓位限制")
            return False
        size_btc = min(size_btc, remaining)
        return await self._place_and_confirm("buy", size_btc, signal, price)

    async def _execute_sell(self, signal: TradeSignal, price: Decimal) -> bool:
        if self._local_position <= 0:
            logger.info("无仓位可卖")
            return False
        return await self._place_and_confirm(
            "sell", self._local_position, signal, price,
        )

    async def _place_and_confirm(self, side: str, size: Decimal, signal: TradeSignal, price: Decimal) -> bool:
        if self._dry_run:
            logger.info(
                f"[DRY-RUN] {side.upper()} {size} {self._ticker} "
                f"conf={signal.confidence:.2f}"
            )
            return True
        if not self._signer or self._market_index is None:
            return False
        idx = int(time.time() * 1_000_000) % 1_000_000_000
        event = asyncio.Event()
        self._pending_fills[idx] = event
        try:
            await place_ioc_order(
                self._signer, self._market_index,
                idx, side, size, self._base_mult,
                self._price_mult, price,
            )
            fill = await wait_for_fill(
                event, self._fill_results, idx, self._fill_timeout,
            )
            if fill:
                self._update_position(side, fill)
                self._trade_count += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Lighter 下单异常: {e}")
            # 等待延迟确认
            try:
                await asyncio.wait_for(event.wait(), SEND_ERROR_GRACE_TIMEOUT)
                if idx in self._fill_results:
                    self._update_position(side, self._fill_results.pop(idx))
                    self._trade_count += 1
                    return True
            except asyncio.TimeoutError:
                pass
            return False
        finally:
            self._pending_fills.pop(idx, None)
            self._fill_results.pop(idx, None)

    def _update_position(self, side: str, fill: dict) -> None:
        filled = fill.get("filled_size", Decimal("0"))
        if side == "buy":
            self._local_position += filled
        else:
            self._local_position = max(self._local_position - filled, Decimal("0"))
        logger.info(f"仓位更新: {side} {filled} → {self._local_position}")

    async def get_position(self) -> Decimal:
        return await query_position(
            self._api_client, self._account_index, self._market_index,
        )

    async def get_balance(self) -> Decimal:
        return await query_balance(self._api_client, self._account_index)

    async def close_all_positions(self) -> bool:
        """平掉所有仓位（reduce_only IOC）。"""
        pos = await self.get_position()
        if abs(pos) == 0:
            return True
        if self._dry_run:
            logger.info(f"[DRY-RUN] 平仓: {pos}")
            return True
        side, idx = "sell" if pos > 0 else "buy", int(time.time() * 1_000_000) % 1_000_000_000
        event = asyncio.Event()
        self._pending_fills[idx] = event
        try:
            # 平仓用最近价格；如果没有则查 REST
            close_price = self._last_price if self._last_price > 0 else await fetch_last_price(self._api_client, self._market_index)
            await place_ioc_order(self._signer, self._market_index, idx, side, abs(pos), self._base_mult, self._price_mult, close_price, reduce_only=True)
            try:
                await asyncio.wait_for(event.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.5)
            remaining = await self.get_position()
            if abs(remaining) < abs(pos) * Decimal("0.5"):
                self._local_position = Decimal("0")
                return True
            return False
        except Exception as e:
            logger.error(f"平仓异常: {e}")
            return False
        finally:
            self._pending_fills.pop(idx, None)

    def get_agent_stats(self, agent_id: str) -> dict:
        return {"agent_id": agent_id, "local_position": float(self._local_position),
                "realized_pnl": float(self._realized_pnl), "trade_count": self._trade_count,
                "consecutive_losses": self._consecutive_losses}
