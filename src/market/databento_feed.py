from __future__ import annotations

"""CME 期货数据源：通过 Databento API 获取实时/历史行情。

Mock 模式复用 MockDataFeed（从 CSV 加载），Live 模式调 Databento SDK。
Databento 文档: https://databento.com/docs
"""

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from loguru import logger

from src.market.data_feed import (
    DataFeed,
    MarketSnapshot,
    MockDataFeed,
    _generate_fake_snapshot,
)

# CME 主要合约默认价格（Mock 回退用）
_CME_DEFAULT_PRICES: dict[str, float] = {
    "ES": 5900.0,   # E-mini S&P 500
    "NQ": 20500.0,  # E-mini Nasdaq 100
    "CL": 70.0,     # 原油
    "GC": 3000.0,   # 黄金
    "SI": 33.0,     # 白银
    "ZB": 112.0,    # 美国国债
}


def create_cme_mock_feed(
    csv_path: str = "",
    asset: str = "ES",
    replay_speed: float = 1.0,
) -> MockDataFeed:
    """创建 CME Mock 数据源（复用 MockDataFeed，注入 CME 默认价格）。"""
    feed = MockDataFeed(csv_path=csv_path, asset=asset, replay_speed=replay_speed)
    return feed


def get_cme_default_price(asset: str) -> float:
    """获取 CME 资产默认价格，未知资产返回 1000.0。"""
    return _CME_DEFAULT_PRICES.get(asset, 1000.0)


class DatabentoCMEFeed(DataFeed):
    """通过 Databento SDK 拉取 CME 期货实时行情。

    需要环境变量 DATABENTO_API_KEY。
    使用 asyncio.to_thread 包装同步 SDK 调用。
    """

    def __init__(
        self,
        dataset: str = "GLBX.MDP3",
        schema: str = "ohlcv-1h",
        interval_seconds: int = 60,
    ) -> None:
        self._api_key: str = os.environ.get("DATABENTO_API_KEY", "")
        self._dataset: str = dataset
        self._schema: str = schema
        self._interval: int = interval_seconds
        self._client = None  # 延迟初始化
        if not self._api_key:
            logger.warning("DATABENTO_API_KEY 未设置，Live 模式不可用")
        logger.info(
            f"DatabentoCMEFeed 初始化: dataset={dataset}, schema={schema}"
        )

    def _ensure_client(self):
        """延迟加载 databento SDK（避免未安装时导入失败）。"""
        if self._client is not None:
            return
        try:
            import databento as db
            self._client = db.Historical(key=self._api_key)
            logger.info("Databento Historical 客户端已初始化")
        except ImportError:
            logger.error("databento 未安装，请运行: pip install databento")
            raise
        except Exception as exc:
            logger.error(f"Databento 客户端初始化失败: {exc}")
            raise

    def _fetch_latest_sync(self, symbol: str) -> dict | None:
        """同步调用 Databento API 获取最新一根 OHLCV。"""
        try:
            self._ensure_client()
        except (ImportError, Exception) as exc:
            logger.warning(f"Databento 客户端不可用: {exc}")
            return None
        try:
            import databento as db
            from datetime import timedelta
            now = datetime.now(tz=timezone.utc)
            start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
            data = self._client.timeseries.get_range(
                dataset=self._dataset,
                symbols=symbol,
                schema=self._schema,
                start=start,
            )
            df = data.to_df()
            if df.empty:
                return None
            # 自动检测 fixed-point：值 > 1e6 说明未转换
            sample = [float(r) for r in df["close"] if float(r) != 0]
            divisor = 1e9 if (sample and sample[0] > 1e6) else 1.0
            row = df.iloc[-1]
            return {
                "open": float(row["open"]) / divisor,
                "high": float(row["high"]) / divisor,
                "low": float(row["low"]) / divisor,
                "close": float(row["close"]) / divisor,
                "volume": int(row["volume"]),
            }
        except Exception as exc:
            logger.error(f"Databento 拉取失败 [{symbol}]: {exc}")
            return None

    async def get_latest(self, asset: str) -> MarketSnapshot | None:
        """异步获取指定 CME 资产最新行情（包装同步 SDK 调用）。"""
        result = await asyncio.to_thread(self._fetch_latest_sync, asset)
        if result is None:
            logger.warning(f"Databento 未获取到 {asset} 数据，使用模拟价格")
            return _generate_fake_snapshot(asset, get_cme_default_price(asset))

        close = result["close"]
        open_p = result["open"]
        change = ((close - open_p) / open_p * 100) if open_p else 0.0
        return MarketSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            asset=asset,
            price=close,
            price_24h_change_pct=round(change, 4),
            volume_24h=float(result["volume"]),
            high_24h=result["high"],
            low_24h=result["low"],
            open_price=open_p,
            funding_rate=0.0,  # CME 期货无资金费率
        )

    async def subscribe(self, assets: list[str]) -> AsyncIterator[MarketSnapshot]:
        """定时轮询 Databento，逐条推送 CME 最新行情。"""
        while True:
            for asset in assets:
                snapshot = await self.get_latest(asset)
                if snapshot is not None:
                    yield snapshot
            await asyncio.sleep(self._interval)
