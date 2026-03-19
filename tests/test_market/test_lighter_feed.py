"""Lighter DEX 行情源测试（Mock SDK，不需要真实连接）。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.market.lighter_feed import LighterLiveDataFeed


class TestLighterLiveDataFeed:
    """LighterLiveDataFeed 单元测试（不连接真实 WS）。"""

    def _make_feed(self) -> LighterLiveDataFeed:
        """创建一个未连接的 feed 实例。"""
        return LighterLiveDataFeed(market_index=0, asset_name="BTC-PERP")

    # ── BBO 与 mid price ──

    def test_get_mid_price_empty_ob(self) -> None:
        """空 orderbook 返回 None。"""
        feed = self._make_feed()
        assert feed.get_mid_price() is None

    def test_get_mid_price_with_data(self) -> None:
        """有 bids/asks 时返回正确的 mid price。"""
        feed = self._make_feed()
        feed._bids = {67000.0: 1.0, 66900.0: 2.0}
        feed._asks = {67100.0: 1.0, 67200.0: 2.0}
        mid = feed.get_mid_price()
        assert mid is not None
        # mid = (67000 + 67100) / 2 = 67050
        assert mid == Decimal("67050")

    def test_is_ready_false_initially(self) -> None:
        """初始状态 is_ready 返回 False。"""
        feed = self._make_feed()
        assert feed.is_ready() is False

    def test_is_ready_true_with_ob(self) -> None:
        """有 OB 数据且 ob_ready=True 时返回 True。"""
        feed = self._make_feed()
        feed._ob_ready = True
        feed._bids = {67000.0: 1.0}
        feed._asks = {67100.0: 1.0}
        assert feed.is_ready() is True

    # ── get_latest ──

    @pytest.mark.asyncio
    async def test_get_latest_empty_ob(self) -> None:
        """空 OB 时 get_latest 返回 None。"""
        feed = self._make_feed()
        snapshot = await feed.get_latest("BTC-PERP")
        assert snapshot is None

    @pytest.mark.asyncio
    async def test_get_latest_with_ob(self) -> None:
        """有 OB 时 get_latest 返回有效 MarketSnapshot。"""
        feed = self._make_feed()
        feed._bids = {67000.0: 1.0}
        feed._asks = {67100.0: 1.0}
        snapshot = await feed.get_latest("BTC-PERP")
        assert snapshot is not None
        assert snapshot.asset == "BTC-PERP"
        # mid = (67000 + 67100) / 2 = 67050
        assert snapshot.price == 67050.0
        assert snapshot.timestamp.tzinfo is not None

    # ── OB 更新处理 ──

    def test_handle_ob_snapshot(self) -> None:
        """OB 快照正确填充 bids/asks。"""
        feed = self._make_feed()
        data = {
            "order_book": {
                "bids": [
                    {"price": "67000", "size": "1.5"},
                    {"price": "66900", "size": "2.0"},
                ],
                "asks": [
                    {"price": "67100", "size": "1.0"},
                    {"price": "67200", "size": "3.0"},
                ],
            }
        }
        feed._handle_ob_snapshot(data)
        assert feed._ob_ready is True
        assert len(feed._bids) == 2
        assert len(feed._asks) == 2
        assert feed._bids[67000.0] == 1.5
        assert feed._asks[67100.0] == 1.0

    def test_handle_ob_update_add_remove(self) -> None:
        """增量更新：添加新 level 和删除已有 level。"""
        feed = self._make_feed()
        feed._ob_ready = True
        feed._bids = {67000.0: 1.0}
        feed._asks = {67100.0: 1.0}
        # 添加新 bid，删除旧 ask（size=0）
        data = {
            "order_book": {
                "bids": [{"price": "66900", "size": "2.0"}],
                "asks": [{"price": "67100", "size": "0"}],
            }
        }
        feed._handle_ob_update(data)
        assert 66900.0 in feed._bids
        assert 67100.0 not in feed._asks

    def test_apply_updates_ignores_invalid(self) -> None:
        """无效数据（缺字段、负价格）被静默跳过。"""
        feed = self._make_feed()
        feed._apply_updates("bids", [
            {"price": "-1", "size": "1"},  # 负价格
            {"size": "1"},                   # 缺 price
            {"price": "abc", "size": "1"},   # 非数字
        ])
        assert len(feed._bids) == 0

    # ── 价格历史 ──

    def test_price_history_tracking(self) -> None:
        """价格历史正确记录，高低价更新。"""
        feed = self._make_feed()
        feed._update_price_history(67000.0)
        feed._update_price_history(67500.0)
        feed._update_price_history(66500.0)
        assert len(feed._price_history) == 3
        assert feed._high_24h == 67500.0
        assert feed._low_24h == 66500.0

    def test_build_snapshot_fields(self) -> None:
        """构建的 snapshot 包含所有必要字段。"""
        feed = self._make_feed()
        feed._update_price_history(67000.0)
        feed._update_price_history(67100.0)  # 更新历史以包含新价格
        snapshot = feed._build_snapshot(67100.0)
        assert snapshot.asset == "BTC-PERP"
        assert snapshot.price == 67100.0
        assert snapshot.high_24h >= 67100.0
        assert snapshot.low_24h <= 67000.0
