from __future__ import annotations

"""Lighter DEX 实时行情数据源，通过 WS 订阅 orderbook 计算 mid price。"""

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from decimal import Decimal

import websockets
from loguru import logger

from src.market.data_feed import DataFeed, MarketSnapshot

WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"
RECONNECT_BASE_DELAY = 1
RECONNECT_MAX_DELAY = 30


class LighterLiveDataFeed(DataFeed):
    """Lighter DEX 实时行情源，基于 WS orderbook BBO。"""

    def __init__(self, market_index: int, asset_name: str = "BTC-PERP") -> None:
        self._market_index = market_index
        self._asset_name = asset_name
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._ws_task: asyncio.Task | None = None
        self._running = False
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._ob_ready = False
        # 价格历史（每分钟一条，保留 24h = 1440 条）
        self._price_history: deque[tuple[float, float]] = deque(maxlen=1440)
        self._high_24h: float = 0.0
        self._low_24h: float = float("inf")

    # ── 连接管理 ──

    async def connect(self) -> None:
        """启动 WS 循环，等待 OB 就绪（最长 10 秒）。"""
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        deadline = time.time() + 10
        while time.time() < deadline:
            if self._ob_ready:
                logger.info("Lighter 行情源就绪")
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("Lighter orderbook 未在 10 秒内就绪")

    async def disconnect(self) -> None:
        """断开 WS 连接。"""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass

    # ── DataFeed 接口 ──

    async def get_latest(self, asset: str) -> MarketSnapshot | None:
        """返回当前 mid price 快照。"""
        bid = max(self._bids.keys()) if self._bids else None
        ask = min(self._asks.keys()) if self._asks else None
        if bid is None or ask is None:
            return None
        mid = (bid + ask) / 2
        self._update_price_history(mid)
        return self._build_snapshot(mid)

    async def subscribe(self, assets: list[str]) -> AsyncIterator[MarketSnapshot]:
        """每秒推送一次行情快照。"""
        while self._running:
            snapshot = await self.get_latest(self._asset_name)
            if snapshot is not None:
                yield snapshot
            await asyncio.sleep(1.0)

    def get_mid_price(self) -> Decimal | None:
        """获取当前中间价（供执行器使用）。"""
        bid = max(self._bids.keys()) if self._bids else None
        ask = min(self._asks.keys()) if self._asks else None
        if bid is None or ask is None:
            return None
        return (Decimal(str(bid)) + Decimal(str(ask))) / 2

    def is_ready(self) -> bool:
        return self._ob_ready and bool(self._bids) and bool(self._asks)

    # ── 价格历史 ──

    def _update_price_history(self, price: float) -> None:
        """更新价格历史和 24h 高低价。"""
        now = time.time()
        self._price_history.append((now, price))
        cutoff = now - 86400
        self._high_24h = price
        self._low_24h = price
        for ts, p in self._price_history:
            if ts >= cutoff:
                self._high_24h = max(self._high_24h, p)
                self._low_24h = min(self._low_24h, p)

    def _build_snapshot(self, price: float) -> MarketSnapshot:
        """构建 MarketSnapshot。"""
        change_24h = 0.0
        if len(self._price_history) > 1:
            cutoff = time.time() - 86400
            for ts, p in self._price_history:
                if ts >= cutoff:
                    change_24h = (price - p) / p * 100 if p else 0.0
                    break
        return MarketSnapshot(
            timestamp=datetime.now(tz=timezone.utc), asset=self._asset_name,
            price=price, price_24h_change_pct=round(change_24h, 4),
            volume_24h=0.0, open_price=price,
            high_24h=self._high_24h if self._high_24h > 0 else price,
            low_24h=self._low_24h if self._low_24h < float("inf") else price,
        )

    # ── WebSocket ──

    async def _ws_loop(self) -> None:
        """WS 主循环，自动重连。"""
        delay = RECONNECT_BASE_DELAY
        while self._running:
            try:
                async with websockets.connect(WS_URL) as ws:
                    self._ws = ws
                    delay = RECONNECT_BASE_DELAY
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channel": f"order_book/{self._market_index}",
                    }))
                    await self._message_loop(ws)
            except Exception as e:
                logger.error(f"Lighter 行情 WS 错误: {e}")
            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def _message_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        while self._running:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            msg_type = data.get("type", "")
            if msg_type == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            elif msg_type == "subscribed/order_book":
                self._handle_ob_snapshot(data)
            elif msg_type == "update/order_book" and self._ob_ready:
                self._handle_ob_update(data)

    def _handle_ob_snapshot(self, data: dict) -> None:
        ob = data.get("order_book", {})
        self._bids.clear()
        self._asks.clear()
        self._apply_updates("bids", ob.get("bids", []))
        self._apply_updates("asks", ob.get("asks", []))
        self._ob_ready = True
        logger.info(f"Lighter OB 快照: {len(self._bids)} bids, {len(self._asks)} asks")

    def _handle_ob_update(self, data: dict) -> None:
        ob = data.get("order_book", {})
        self._apply_updates("bids", ob.get("bids", []))
        self._apply_updates("asks", ob.get("asks", []))

    def _apply_updates(self, side: str, updates: list) -> None:
        ob = self._bids if side == "bids" else self._asks
        for u in updates:
            try:
                price, size = float(u["price"]), float(u["size"])
                if price <= 0:
                    continue
                if size == 0:
                    ob.pop(price, None)
                elif size > 0:
                    ob[price] = size
            except (KeyError, ValueError, TypeError):
                continue
