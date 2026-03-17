from __future__ import annotations
"""资产匿名化器：防止 LLM 通过资产名称回忆历史走势。

原理：将 "BTC-PERP" → "ASSET_A"，"ETH-PERP" → "ASSET_B" 等。
Prompt 生成阶段替换，信号解析阶段反向替换。
"""

_LABELS = [f"ASSET_{chr(65 + i)}" for i in range(26)]  # ASSET_A ~ ASSET_Z


class AssetAnonymizer:
    """双向资产名称映射器。"""

    def __init__(self, asset_list: list[str]) -> None:
        """建立双向映射表。"""
        self._real_to_anon: dict[str, str] = {}
        self._anon_to_real: dict[str, str] = {}
        for i, asset in enumerate(asset_list):
            label = _LABELS[i] if i < len(_LABELS) else f"ASSET_{i}"
            self._real_to_anon[asset] = label
            self._anon_to_real[label] = asset

    def anonymize(self, text: str) -> str:
        """将所有已知资产名替换为匿名标签。"""
        result = text
        for real, anon in self._real_to_anon.items():
            result = result.replace(real, anon)
        return result

    def deanonymize(self, text: str) -> str:
        """将匿名标签还原为真实资产名。"""
        result = text
        for anon, real in self._anon_to_real.items():
            result = result.replace(anon, real)
        return result

    def anonymize_market_data(self, data: dict) -> dict:
        """匿名化行情数据字典中的 asset 字段。"""
        result = data.copy()
        if "asset" in result:
            result["asset"] = self._real_to_anon.get(result["asset"], result["asset"])
        return result

    def deanonymize_asset(self, asset: str) -> str:
        """单个资产名反匿名化。"""
        return self._anon_to_real.get(asset, asset)

    def anonymize_asset(self, asset: str) -> str:
        """单个资产名匿名化。"""
        return self._real_to_anon.get(asset, asset)
