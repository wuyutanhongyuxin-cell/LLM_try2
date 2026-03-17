from __future__ import annotations

"""资产匿名化器测试。"""

import pytest

from src.utils.anonymizer import AssetAnonymizer

ASSETS = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]


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
