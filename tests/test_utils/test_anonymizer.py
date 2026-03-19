from __future__ import annotations

"""资产匿名化器测试。"""

import pytest

from src.utils.anonymizer import AssetAnonymizer

ASSETS = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
CME_ASSETS = ["ES", "NQ", "CL", "GC", "SI", "ZB"]


@pytest.fixture()
def anon() -> AssetAnonymizer:
    """创建匿名化器实例。"""
    return AssetAnonymizer(ASSETS)


class TestAnonymize:
    """匿名化测试。"""

    def test_anonymize_replaces_asset(self, anon: AssetAnonymizer) -> None:
        """BTC-PERP 应被替换为 ASSET_A。"""
        result = anon.anonymize("BTC-PERP is trending")
        assert "ASSET_A" in result
        assert "BTC-PERP" not in result

    def test_deanonymize_restores(self, anon: AssetAnonymizer) -> None:
        """ASSET_A 应被还原为 BTC-PERP。"""
        result = anon.deanonymize("Buy ASSET_A at 67000")
        assert "BTC-PERP" in result
        assert "ASSET_A" not in result

    def test_roundtrip(self, anon: AssetAnonymizer) -> None:
        """anonymize -> deanonymize 应恢复原始文本。"""
        original = "BTC-PERP up 3%, ETH-PERP down 1%, SOL-PERP flat"
        anonymized = anon.anonymize(original)
        restored = anon.deanonymize(anonymized)
        assert restored == original

    def test_anonymize_market_data(self, anon: AssetAnonymizer) -> None:
        """dict 中 asset 字段应被替换为匿名标签。"""
        data = {"asset": "ETH-PERP", "price": 3500.0}
        result = anon.anonymize_market_data(data)
        assert result["asset"] == "ASSET_B"
        assert result["price"] == 3500.0

    def test_unknown_asset_unchanged(self, anon: AssetAnonymizer) -> None:
        """未注册的资产名不应被修改。"""
        result = anon.anonymize("DOGE-PERP is mooning")
        assert "DOGE-PERP" in result

    def test_no_real_names_in_anonymized(self, anon: AssetAnonymizer) -> None:
        """匿名化后的文本中不应包含任何真实资产名。"""
        text = "BTC-PERP ETH-PERP SOL-PERP analysis"
        anonymized = anon.anonymize(text)
        for asset in ASSETS:
            assert asset not in anonymized

    def test_multiple_assets(self, anon: AssetAnonymizer) -> None:
        """多个资产应同时被正确匿名化。"""
        text = "BTC-PERP vs ETH-PERP vs SOL-PERP"
        anonymized = anon.anonymize(text)
        assert "ASSET_A" in anonymized
        assert "ASSET_B" in anonymized
        assert "ASSET_C" in anonymized


class TestCMEAnonymize:
    """CME 品种匿名化测试——验证短名称不误替换指标名。"""

    @pytest.fixture()
    def cme_anon(self) -> AssetAnonymizer:
        """CME 品种匿名化器。"""
        return AssetAnonymizer(CME_ASSETS)

    def test_si_does_not_break_rsi(self, cme_anon: AssetAnonymizer) -> None:
        """SI（白银）的替换不应破坏 RSI 指标名。"""
        text = "RSI(14) = 35.2, oversold signal on SI"
        result = cme_anon.anonymize(text)
        assert "RSI(14)" in result, f"RSI 被误替换: {result}"
        assert "SI" not in result.replace("RSI", "")  # 独立的 SI 应被替换

    def test_es_does_not_break_prices(self, cme_anon: AssetAnonymizer) -> None:
        """ES 不应破坏 prices、closes 等英文单词。"""
        text = "ES closes at 5900, resistance level"
        result = cme_anon.anonymize(text)
        assert "closes" in result, f"closes 被误替换: {result}"

    def test_cl_does_not_break_close(self, cme_anon: AssetAnonymizer) -> None:
        """CL 不应破坏 close 等英文单词。"""
        text = "CL is near close price 72.50"
        result = cme_anon.anonymize(text)
        assert "close" in result, f"close 被误替换: {result}"

    def test_cme_roundtrip(self, cme_anon: AssetAnonymizer) -> None:
        """CME 品种匿名化→反匿名化应恢复原文中出现的资产名。"""
        text = "Buy ES at 5900, sell GC at 2400, SI at 28"
        present_assets = ["ES", "GC", "SI"]
        anonymized = cme_anon.anonymize(text)
        for asset in present_assets:
            assert asset not in anonymized, f"{asset} 未被匿名化"
        restored = cme_anon.deanonymize(anonymized)
        for asset in present_assets:
            assert asset in restored, f"{asset} 未被还原"

    def test_rsi_sma_macd_preserved(self, cme_anon: AssetAnonymizer) -> None:
        """技术指标名 RSI、SMA、MACD 不应被任何 CME 品种替换破坏。"""
        text = "RSI=28 MACD bullish SMA(20)=5850 ES trading"
        result = cme_anon.anonymize(text)
        assert "RSI=28" in result
        assert "MACD" in result
        assert "SMA(20)" in result
        assert "ES" not in result  # ES 应被替换
