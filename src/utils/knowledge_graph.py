"""轻量级市场知识图谱加载与查询。

从 config/market_knowledge.json 加载因果关系图谱，
提供查询接口供 Decision Prompt 注入上下文。
不引入任何新依赖（纯 JSON + 标准库）。
"""
from __future__ import annotations

import json
from pathlib import Path

# 图谱文件路径（项目根目录/config/market_knowledge.json）
_GRAPH_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "market_knowledge.json"

# 内存缓存，避免重复读取文件
_graph_cache: dict | None = None

# strength 排序权重
_STRENGTH_ORDER: dict[str, int] = {"strong": 0, "moderate": 1, "weak": 2}


def load_graph() -> dict:
    """加载知识图谱，带内存缓存。首次调用读文件，后续直接返回缓存。"""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache
    with open(_GRAPH_PATH, "r", encoding="utf-8") as f:
        _graph_cache = json.load(f)
    return _graph_cache


def get_causal_factors(asset: str) -> list[dict]:
    """查询影响指定资产的所有因果因子。

    Args:
        asset: 资产标识，如 "BTC"

    Returns:
        影响该资产的因果关系列表，按 strength 排序（strong > moderate > weak）
    """
    graph = load_graph()
    relations = graph.get("causal_relations", [])
    # 筛选 to == asset 的关系
    factors = [r for r in relations if r.get("to") == asset]
    # 按 strength 排序
    factors.sort(key=lambda r: _STRENGTH_ORDER.get(r.get("strength", ""), 99))
    return factors


def get_regime_context(market_signals: dict[str, float] | None = None) -> str:
    """根据当前市场信号判断所处 regime 并返回自然语言描述。

    Args:
        market_signals: 可选的市场信号字典，如 {"funding_rate": 0.0005}
            当前阶段此参数可为 None（仅返回通用 regime 知识）

    Returns:
        英文自然语言段落，可直接插入 Decision Prompt
    """
    graph = load_graph()
    indicators = graph.get("regime_indicators", [])
    lines: list[str] = []
    for ind in indicators:
        cond = ind.get("condition", "")
        impl = ind.get("implication", "")
        lines.append(f"- If {cond}: {impl}")
    return "\n".join(lines)


def build_knowledge_context(asset: str) -> str:
    """构建指定资产的完整知识上下文字符串，用于注入 Prompt。

    格式:
    === MARKET KNOWLEDGE ===
    Causal factors affecting {asset}:
    - M2_SUPPLY (positive, strong, ~90d lag): ...
    Regime indicators:
    - If funding rate > 0.03%: ...
    """
    factors = get_causal_factors(asset)
    lines: list[str] = [f"=== MARKET KNOWLEDGE ==="]
    # 因果因子段
    lines.append(f"Causal factors affecting {asset}:")
    if factors:
        for f in factors:
            src = f.get("from", "?")
            effect = f.get("effect", "?")
            strength = f.get("strength", "?")
            lag = f.get("lag_days", 0)
            mech = f.get("mechanism", "")
            lines.append(f"- {src} ({effect}, {strength}, ~{lag}d lag): {mech}")
    else:
        lines.append("- No known causal factors for this asset")
    # Regime 段
    lines.append("")
    lines.append("Regime indicators:")
    regime_text = get_regime_context()
    lines.append(regime_text)
    return "\n".join(lines)
