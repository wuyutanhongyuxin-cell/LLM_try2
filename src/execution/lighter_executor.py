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
    fetch_last_price, place_ioc_order, query_balance, query_position,
)
from src.execution.signal import Action, TradeSignal

BASE_URL = "https://mainnet.zklighter.elliot.ai"


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
            return await self._execute_sell(signal, balance, current_price)
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

    async def _execute_sell(
        self, signal: TradeSignal, balance: Decimal, price: Decimal,
    ) -> bool:
        """SELL = 平多仓 或 开空仓（永续合约支持做空）。"""
        # 可卖额度 = max_position + 当前仓位（正=多仓可卖+开空，负=已空）
        remaining = self._max_position + self._local_position
        if remaining <= 0:
            logger.warning("已达最大空仓限制")
            return False
        size_usd = balance * Decimal(str(signal.size_pct)) / Decimal("100")
        size_btc = size_usd / price
        size_btc = min(size_btc, remaining)
        return await self._place_and_confirm("sell", size_btc, signal, price)

    async def _place_and_confirm(self, side: str, size: Decimal, signal: TradeSignal, price: Decimal) -> bool:
        if self._dry_run:
            logger.info(
                f"[DRY-RUN] {side.upper()} {size} {self._ticker} "
                f"conf={signal.confidence:.2f}"
            )
            return True
        if not self._signer or self._market_index is None:
            return False
        pos_before = await self.get_position()
        idx = int(time.time() * 1_000_000) % 1_000_000_000
        try:
            await place_ioc_order(
                self._signer, self._market_index,
                idx, side, size, self._base_mult,
                self._price_mult, price,
            )
        except Exception as e:
            logger.error(f"Lighter 下单异常: {e}")
        # 等待 Lighter 处理，然后用 REST 查询仓位确认成交
        await asyncio.sleep(1.5)
        pos_after = await self.get_position()
        filled = abs(pos_after - pos_before)
        if filled > Decimal("0.000001"):
            self._local_position = pos_after
            self._trade_count += 1
            logger.info(
                f"REST 确认成交: {side} {filled} {self._ticker} "
                f"仓位 {pos_before}→{pos_after}"
            )
            return True
        logger.warning(f"REST 确认未成交: 仓位未变 {pos_before}")
        return False

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
        side = "sell" if pos > 0 else "buy"
        idx = int(time.time() * 1_000_000) % 1_000_000_000
        try:
            close_price = self._last_price if self._last_price > 0 else await fetch_last_price(self._api_client, self._market_index)
            await place_ioc_order(self._signer, self._market_index, idx, side, abs(pos), self._base_mult, self._price_mult, close_price, reduce_only=True)
            await asyncio.sleep(1.5)
            remaining = await self.get_position()
            self._local_position = remaining
            if abs(remaining) < abs(pos) * Decimal("0.5"):
                logger.info(f"平仓成功: {pos}→{remaining}")
                return True
            logger.warning(f"平仓部分成交: {pos}→{remaining}")
            return False
        except Exception as e:
            logger.error(f"平仓异常: {e}")
            return False

    def get_agent_stats(self, agent_id: str) -> dict:
        return {"agent_id": agent_id, "local_position": float(self._local_position),
                "realized_pnl": float(self._realized_pnl), "trade_count": self._trade_count,
                "consecutive_losses": self._consecutive_losses}
