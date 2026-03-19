from __future__ import annotations
"""资产匿名化器：防止 LLM 通过资产名称回忆历史走势。

原理：将 "BTC-PERP" → "ASSET_A"，"ETH-PERP" → "ASSET_B" 等。
Prompt 生成阶段替换，信号解析阶段反向替换。
使用单词边界正则避免误替换（如 "SI" 不会破坏 "RSI"）。
"""

import re

_LABELS = [f"ASSET_{chr(65 + i)}" for i in range(26)]  # ASSET_A ~ ASSET_Z


class AssetAnonymizer:
    """双向资产名称映射器（单词边界安全替换）。"""

    def __init__(self, asset_list: list[str]) -> None:
        """建立双向映射表和正则模式。"""
        self._real_to_anon: dict[str, str] = {}
        self._anon_to_real: dict[str, str] = {}
        # 按长度降序排列，确保长名称优先匹配（如 BTC-PERP 优先于 BTC）
        self._sorted_real: list[str] = sorted(asset_list, key=len, reverse=True)
        for i, asset in enumerate(asset_list):
            label = _LABELS[i] if i < len(_LABELS) else f"ASSET_{i}"
            self._real_to_anon[asset] = label
            self._anon_to_real[label] = asset
        # 预编译正则：使用单词边界 \b 避免 "SI" 误替换 "RSI" 中的 SI
        self._anon_pattern = re.compile(
            "|".join(re.escape(a) for a in self._sorted_real
                     if a in self._real_to_anon),
            flags=re.IGNORECASE,
        )

    def _replace_match(self, match: re.Match) -> str:
        """正则回调：将匹配到的资产名替换为匿名标签。"""
        return self._real_to_anon.get(match.group(0), match.group(0))

    def anonymize(self, text: str) -> str:
        """将所有已知资产名替换为匿名标签（单词边界安全）。"""
        result = text
        # 按长度降序逐个替换，使用 \b 单词边界
        for real in self._sorted_real:
            anon = self._real_to_anon[real]
            # \b 单词边界：确保 "SI" 只匹配独立的 "SI"，不匹配 "RSI"
            pattern = r'\b' + re.escape(real) + r'\b'
            result = re.sub(pattern, anon, result)
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
