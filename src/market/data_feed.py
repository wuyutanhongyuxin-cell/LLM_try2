from __future__ import annotations

"""
行情数据源抽象层。

提供统一的行情数据接口，支持两种实现：
- MockDataFeed：从 CSV 加载历史数据，或生成模拟随机行情
- LiveDataFeed：通过 Binance REST API 拉取实时行情
"""
import asyncio
import csv
import os
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    """单个时间点的行情快照。"""
    timestamp: datetime = Field(..., description="快照时间戳")
    asset: str = Field(..., description="资产标识，如 BTC-PERP")
    price: float = Field(..., description="当前价格")
    price_24h_change_pct: float = Field(..., description="24小时涨跌幅百分比")
    volume_24h: float = Field(..., description="24小时成交量")
    high_24h: float = Field(..., description="24小时最高价")
    low_24h: float = Field(..., description="24小时最低价")
    open_price: float = Field(default=0.0, description="开盘价（回测需要）")
    funding_rate: float = Field(default=0.0, description="当前资金费率（永续合约）")


class DataFeed(ABC):
    """行情数据源抽象基类。"""

    @abstractmethod
    async def get_latest(self, asset: str) -> MarketSnapshot | None: ...

    @abstractmethod
    async def subscribe(self, assets: list[str]) -> AsyncIterator[MarketSnapshot]: ...


# --------------- 辅助函数 ---------------

def _parse_csv_row(
    row: dict[str, str], asset: str, change_24h_override: float | None = None
) -> MarketSnapshot:
    """将 CSV 行解析为 MarketSnapshot。change_24h_override 可由外部传入精确 24h 变化。"""
    close, open_p = float(row["close"]), float(row["open"])
    change = change_24h_override if change_24h_override is not None else (
        ((close - open_p) / open_p * 100) if open_p else 0.0
    )
    return MarketSnapshot(
        timestamp=datetime.fromisoformat(row["timestamp"]), asset=asset,
        price=close, price_24h_change_pct=round(change, 4),
        volume_24h=float(row["volume"]),
        high_24h=float(row["high"]), low_24h=float(row["low"]),
        open_price=open_p,
    )


def _generate_fake_snapshot(asset: str, base_price: float) -> MarketSnapshot:
    """生成一条模拟随机行情（无 CSV 文件时的回退方案）。"""
    pct = random.uniform(-5.0, 5.0)
    price = base_price * (1 + pct / 100)
    return MarketSnapshot(
        timestamp=datetime.now(tz=timezone.utc), asset=asset,
        price=round(price, 2), price_24h_change_pct=round(pct, 4),
        volume_24h=round(random.uniform(1000, 100000), 2),
        high_24h=round(price * random.uniform(1.0, 1.03), 2),
        low_24h=round(price * random.uniform(0.97, 1.0), 2),
        open_price=round(base_price, 2),
    )


def _load_csv(csv_path: str) -> list[dict[str, str]]:
    """加载 CSV 文件，返回行列表。不存在则返回空列表。"""
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"CSV 文件不存在: {csv_path}，将使用模拟数据")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _asset_to_binance_symbol(asset: str) -> str:
    """资产名转换：BTC-PERP → BTCUSDT"""
    return asset.replace("-PERP", "USDT")


# 各市场资产默认价格（Mock 回退用）
_DEFAULT_PRICES: dict[str, float] = {
    # 加密货币永续合约
    "BTC-PERP": 67000.0, "ETH-PERP": 3500.0,
    "SOL-PERP": 150.0, "ARB-PERP": 1.2, "DOGE-PERP": 0.15,
    # CME 期货
    "ES": 5900.0, "NQ": 20500.0, "CL": 70.0,
    "GC": 3000.0, "SI": 33.0, "ZB": 112.0,
}


# --------------- MockDataFeed ---------------

class MockDataFeed(DataFeed):
    """模拟数据源：从 CSV 加载历史数据，或生成随机行情。"""

    def __init__(self, csv_path: str = "", asset: str = "BTC-PERP",
                 replay_speed: float = 1.0,
                 adversarial_scenarios: list[tuple[str, int]] | None = None) -> None:
        self._rows: list[dict[str, str]] = _load_csv(csv_path) if csv_path else []
        self._asset = asset
        self._index = 0
        self._replay_speed = replay_speed
        self._price_history: list[float] = []  # 维护价格历史用于精确 24h 变化
        # 对抗性场景注入：修改 _rows 中的价格列
        if adversarial_scenarios and self._rows:
            self._inject_adversarial_scenarios(adversarial_scenarios)
        logger.info(f"MockDataFeed 初始化: {len(self._rows)} 条历史数据, 资产={asset}")

    def _inject_adversarial_scenarios(
        self, scenarios: list[tuple[str, int]]
    ) -> None:
        """将对抗性场景注入到 CSV 数据的价格列中。只修改价格，不改 timestamp/volume。"""
        from src.market.adversarial import generate_adversarial_prices
        for scenario_name, inject_at in scenarios:
            if inject_at >= len(self._rows):
                continue
            base_price = float(self._rows[inject_at]["close"])
            adv_prices = generate_adversarial_prices(base_price, scenario_name)  # type: ignore[arg-type]
            for i, price in enumerate(adv_prices):
                pos = inject_at + i
                if pos >= len(self._rows):
                    break
                self._rows[pos]["close"] = str(price)
                self._rows[pos]["high"] = str(max(price, float(self._rows[pos]["high"])))
                self._rows[pos]["low"] = str(min(price, float(self._rows[pos]["low"])))
                self._rows[pos]["open"] = str(price * 1.001)  # 微调 open 保持合理
            logger.info(f"对抗性场景 '{scenario_name}' 注入到位置 {inject_at}")

    async def get_latest(self, asset: str) -> MarketSnapshot | None:
        """返回当前位置的行情数据。用 24 条前价格计算真实 24h 变化。"""
        if not self._rows:
            return _generate_fake_snapshot(asset, _DEFAULT_PRICES.get(asset, 1000.0))
        if self._index >= len(self._rows):
            self._index = 0
        row = self._rows[self._index]
        close = float(row["close"])
        self._price_history.append(close)
        # 用 24 条前的价格计算 24h 变化（1h K 线 × 24 = 24h）
        change_24h: float | None = None
        if len(self._price_history) > 24:
            price_24h_ago = self._price_history[-25]
            change_24h = (close - price_24h_ago) / price_24h_ago * 100
        snapshot = _parse_csv_row(row, asset, change_24h_override=change_24h)
        self._index += 1
        return snapshot

    async def subscribe(self, assets: list[str]) -> AsyncIterator[MarketSnapshot]:
        """按时间顺序逐条推送，支持加速回放。"""
        if not self._rows:
            while True:  # 无 CSV：持续生成随机数据
                for a in assets:
                    yield _generate_fake_snapshot(a, _DEFAULT_PRICES.get(a, 1000.0))
                await asyncio.sleep(1.0 / self._replay_speed)
        else:
            for row in self._rows:
                for a in assets:
                    yield _parse_csv_row(row, a)
                await asyncio.sleep(0.1 / self._replay_speed)


# --------------- LiveDataFeed ---------------

class LiveDataFeed(DataFeed):
    """实时数据源：通过 Binance REST API 拉取公开行情。"""

    def __init__(self, base_url: str = "", interval_seconds: int = 60) -> None:
        self._base_url = base_url or os.environ.get(
            "BINANCE_BASE_URL", "https://api.binance.com")
        self._interval = interval_seconds
        self._session: aiohttp.ClientSession | None = None
        logger.info(f"LiveDataFeed 初始化: base_url={self._base_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """懒加载 aiohttp session。"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_latest(self, asset: str) -> MarketSnapshot | None:
        """从 Binance 拉取指定资产的 24hr ticker 数据。"""
        symbol = _asset_to_binance_symbol(asset)
        url = f"{self._base_url}/api/v3/ticker/24hr?symbol={symbol}"
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.error(f"Binance API 返回 {resp.status}: {asset}")
                    return None
                data = await resp.json()
            return MarketSnapshot(
                timestamp=datetime.now(tz=timezone.utc), asset=asset,
                price=float(data["lastPrice"]),
                price_24h_change_pct=float(data["priceChangePercent"]),
                volume_24h=float(data["volume"]),
                high_24h=float(data["highPrice"]), low_24h=float(data["lowPrice"]),
                open_price=float(data.get("openPrice", 0)),
            )
        except Exception as e:
            logger.error(f"拉取行情失败 [{asset}]: {e}")
            return None

    async def subscribe(self, assets: list[str]) -> AsyncIterator[MarketSnapshot]:
        """定时轮询 Binance，逐条推送最新行情。"""
        while True:
            for asset in assets:
                snapshot = await self.get_latest(asset)
                if snapshot is not None:
                    yield snapshot
            await asyncio.sleep(self._interval)

    async def close(self) -> None:
        """关闭 HTTP 会话。"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("LiveDataFeed HTTP 会话已关闭")
