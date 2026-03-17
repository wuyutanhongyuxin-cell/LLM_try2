"""Databento CME 数据源测试。"""
from __future__ import annotations

import pytest

from src.market.databento_feed import (
    DatabentoCMEFeed,
    create_cme_mock_feed,
    get_cme_default_price,
)


class TestGetCMEDefaultPrice:
    """CME 默认价格测试。"""

    def test_es_default(self) -> None:
        """ES 默认价格 5900。"""
        assert get_cme_default_price("ES") == 5900.0

    def test_cl_default(self) -> None:
        """CL 默认价格 70。"""
        assert get_cme_default_price("CL") == 70.0

    def test_gc_default(self) -> None:
        """GC 默认价格 3000。"""
        assert get_cme_default_price("GC") == 3000.0

    def test_zb_default(self) -> None:
        """ZB 默认价格 112。"""
        assert get_cme_default_price("ZB") == 112.0

    def test_nq_default(self) -> None:
        """NQ 默认价格 20500。"""
        assert get_cme_default_price("NQ") == 20500.0

    def test_si_default(self) -> None:
        """SI 默认价格 33。"""
        assert get_cme_default_price("SI") == 33.0

    def test_unknown_returns_1000(self) -> None:
        """未知资产返回 1000.0。"""
        assert get_cme_default_price("UNKNOWN") == 1000.0


class TestCreateCMEMockFeed:
    """CME Mock 数据源创建测试。"""

    def test_returns_mock_feed(self) -> None:
        """无 CSV 时返回 MockDataFeed 实例（不崩溃）。"""
        from src.market.data_feed import MockDataFeed
        feed = create_cme_mock_feed(asset="ES")
        assert isinstance(feed, MockDataFeed)

    @pytest.mark.asyncio
    async def test_mock_feed_with_csv(self, tmp_path) -> None:
        """有 CSV 时正确加载数据。"""
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            "2025-09-15 00:00:00,5900.00,5910.00,5890.00,5905.00,3000\n"
            "2025-09-15 01:00:00,5905.00,5920.00,5900.00,5915.00,2500\n"
        )
        csv_file = tmp_path / "es_test.csv"
        csv_file.write_text(csv_content)
        feed = create_cme_mock_feed(csv_path=str(csv_file), asset="ES")
        snapshot = await feed.get_latest("ES")
        assert snapshot is not None
        assert snapshot.asset == "ES"
        assert 5800 < snapshot.price < 6000


class TestDatabentoCMEFeed:
    """Databento Live 数据源测试（无 API key 场景）。"""

    def test_init_without_api_key(self, monkeypatch) -> None:
        """无 DATABENTO_API_KEY 时初始化不崩溃。"""
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        feed = DatabentoCMEFeed()
        assert feed._api_key == ""

    @pytest.mark.asyncio
    async def test_get_latest_fallback(self, monkeypatch) -> None:
        """无 API key 时 get_latest 回退到模拟数据。"""
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        feed = DatabentoCMEFeed()
        # _fetch_latest_sync 会因为没有 API key 而失败，回退到 fake snapshot
        snapshot = await feed.get_latest("ES")
        assert snapshot is not None
        assert snapshot.asset == "ES"
        assert snapshot.funding_rate == 0.0  # CME 无资金费率
