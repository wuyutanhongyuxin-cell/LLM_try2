from __future__ import annotations

"""Prompt 生成器：将 OCEAN 人格参数和交易约束转化为 LLM Prompt。

支持多市场类型：crypto（加密货币永续合约）和 cme（CME 期货）。
Prompt 全部用英文（LLM 英文推理更准），日志和通知用中文。
末尾附带 SHA256 前 12 位 hash 用于版本追溯。
"""

import hashlib

from src.personality.ocean_model import OceanProfile
from src.personality.trait_to_constraint import TradingConstraints
from src.utils.knowledge_graph import build_knowledge_context

# ── 维度描述映射：(全名, 高分含义, 低分含义) ──
# 按市场类型区分：crypto 和 cme 有不同的高/低分描述
_TRAIT_DESC_CRYPTO: dict[str, tuple[str, str, str]] = {
    "openness": (
        "Openness",
        "explores new altcoins, novel strategies, and high-volatility assets",
        "sticks to BTC/ETH only, conservative and proven strategies",
    ),
    "conscientiousness": (
        "Conscientiousness",
        "strict stop-loss discipline, rigorous position sizing, rule-following",
        "impulsive trading, may ignore risk management rules",
    ),
    "extraversion": (
        "Extraversion",
        "follows market sentiment, momentum-chasing, trend-following",
        "contrarian, independent judgment, fades the crowd",
    ),
    "agreeableness": (
        "Agreeableness",
        "herding behavior, aligns with market consensus",
        "challenges consensus, comfortable taking the opposite side",
    ),
    "neuroticism": (
        "Neuroticism",
        "extreme loss aversion, very tight stops, frequent cutting of losers",
        "emotionally stable, can hold through drawdowns patiently",
    ),
}

_TRAIT_DESC_CME: dict[str, tuple[str, str, str]] = {
    "openness": (
        "Openness",
        "trades diverse futures (energy, metals, bonds), embraces cross-asset strategies",
        "sticks to equity index futures (ES/NQ) only, conservative approach",
    ),
    "conscientiousness": (
        "Conscientiousness",
        "strict stop-loss discipline, rigorous position sizing, rule-following",
        "impulsive trading, may ignore risk management rules",
    ),
    "extraversion": (
        "Extraversion",
        "follows institutional flow, momentum-chasing, trend-following",
        "contrarian, independent judgment, fades the crowd",
    ),
    "agreeableness": (
        "Agreeableness",
        "herding behavior, aligns with market consensus and COT data",
        "challenges consensus, comfortable taking the opposite side",
    ),
    "neuroticism": (
        "Neuroticism",
        "extreme loss aversion, very tight stops, frequent cutting of losers",
        "emotionally stable, can hold through drawdowns patiently",
    ),
}

# 市场类型 → 描述映射
_TRAIT_DESCS: dict[str, dict[str, tuple[str, str, str]]] = {
    "crypto": _TRAIT_DESC_CRYPTO,
    "cme": _TRAIT_DESC_CME,
}

# 市场类型 → 角色描述
_ROLE_TEMPLATES: dict[str, str] = {
    "crypto": (
        "You are a cryptocurrency perpetual futures trader with a distinct personality. "
        "Your trading persona is '{name}'. "
        "Your personality directly shapes how you analyze markets and make decisions."
    ),
    "cme": (
        "You are a CME futures trader with a distinct personality. "
        "You trade instruments like E-mini S&P 500 (ES), Nasdaq 100 (NQ), "
        "Crude Oil (CL), Gold (GC), and other CME Globex contracts. "
        "Your trading persona is '{name}'. "
        "Your personality directly shapes how you analyze markets and make decisions."
    ),
}


def _build_personality_section(
    profile: OceanProfile, market_type: str = "crypto"
) -> str:
    """段落2：逐维度列出分数及交易行为含义。"""
    trait_desc = _TRAIT_DESCS.get(market_type, _TRAIT_DESC_CRYPTO)
    lines: list[str] = ["## Your Personality Profile (Big Five / OCEAN)"]
    for attr, (name, high, low) in trait_desc.items():
        score: int = getattr(profile, attr)
        desc = high if score > 50 else low
        lines.append(f"- {name} ({attr[0].upper()}={score}): {desc}")
    return "\n".join(lines)


def _build_constraints_section(c: TradingConstraints) -> str:
    """段落3：注入硬约束，LLM 不得超出。"""
    lines: list[str] = ["## HARD CONSTRAINTS (you MUST NOT exceed these):"]
    for name, info in c.model_fields.items():
        lines.append(f"- {info.description or name}: {getattr(c, name)}")
    return "\n".join(lines)


def generate_system_prompt(
    profile: OceanProfile,
    constraints: TradingConstraints,
    market_type: str = "crypto",
) -> str:
    """生成 System Prompt（英文），Agent 初始化时调用一次。

    Args:
        profile: OCEAN 人格配置
        constraints: 交易硬约束
        market_type: 市场类型 "crypto" 或 "cme"
    """
    role_tmpl = _ROLE_TEMPLATES.get(market_type, _ROLE_TEMPLATES["crypto"])
    role = role_tmpl.format(name=profile.name)
    personality = _build_personality_section(profile, market_type)
    hard = _build_constraints_section(constraints)
    # 资产示例根据市场类型调整
    asset_example = '"BTC-PERP"' if market_type == "crypto" else '"ES"'
    output_fmt = (
        "## Output Format\n"
        "You must respond with a single JSON object containing these fields:\n"
        '- "action": "BUY" | "SELL" | "HOLD"\n'
        f'- "asset": string (e.g. {asset_example})\n'
        '- "size_pct": float (percentage of portfolio)\n'
        '- "entry_price": float\n'
        '- "stop_loss_price": float\n'
        '- "take_profit_price": float\n'
        '- "confidence": float between 0 and 1\n'
        '- "reasoning": string (your analysis)\n'
        '- "personality_influence": string (which trait dominated this decision)'
    )
    rules = (
        "## Rules\n"
        "- Do NOT fabricate market data or prices.\n"
        "- Do NOT exceed any hard constraint listed above.\n"
        "- Do NOT output anything other than the JSON object.\n"
        "- Do NOT wrap the JSON in markdown code fences.\n"
        "You MUST respond with ONLY a valid JSON object."
    )
    prompt = "\n\n".join([role, personality, hard, output_fmt, rules])
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    prompt += f"\n\n[prompt_version: {prompt_hash}]"
    return prompt


def get_prompt_hash(prompt: str) -> str:
    """从 system prompt 末尾提取 hash，或对整体计算 hash。"""
    marker = "[prompt_version: "
    idx = prompt.rfind(marker)
    if idx >= 0:
        return prompt[idx + len(marker):idx + len(marker) + 12]
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def generate_decision_prompt(
    market_data: dict,
    positions: list,
    memory_context: str,
    portfolio_value: float,
) -> str:
    """生成 Decision Prompt（英文），每次决策循环调用。"""
    # 行情
    asset = market_data.get("asset", "UNKNOWN")
    price = market_data.get("price", 0)
    change = market_data.get("change_24h", 0)
    volume = market_data.get("volume", 0)
    market_sec = (
        f"## Current Market Data\n"
        f"- Asset: {asset}\n"
        f"- Price: ${price:,.2f}\n"
        f"- 24h Change: {change:+.2f}%\n"
        f"- 24h Volume: ${volume:,.0f}"
    )
    # 持仓
    if positions:
        pl = ["## Current Positions"]
        for p in positions:
            pl.append(
                f"- {p.get('asset', '?')}: size={p.get('size', 0)}, "
                f"entry=${p.get('entry_price', 0):,.2f}, "
                f"unrealized_pnl=${p.get('unrealized_pnl', 0):,.2f}"
            )
        pos_sec = "\n".join(pl)
    else:
        pos_sec = "## Current Positions\nNo open positions."
    # 资产
    used = sum(p.get("size", 0) * p.get("entry_price", 0) for p in positions)
    port_sec = (
        f"## Portfolio\n"
        f"- Total Value: ${portfolio_value:,.2f}\n"
        f"- Available Balance: ${portfolio_value - used:,.2f}"
    )
    # 知识图谱上下文（在记忆之前注入）
    # 从资产标识提取知识图谱 key：BTC-PERP → BTC，ES → ES
    raw_asset = market_data.get("asset", "UNKNOWN")
    asset_name = raw_asset.replace("-PERP", "") if "-" in raw_asset else raw_asset
    knowledge_sec = f"## Market Knowledge\n{build_knowledge_context(asset_name)}"
    # 记忆
    mem_sec = f"## Memory & Context\n{memory_context}" if memory_context else ""
    # 组装
    instruction = (
        "Based on the above data and your personality, decide your next action. "
        "Respond with ONLY a valid JSON object."
    )
    parts = [market_sec, pos_sec, port_sec, knowledge_sec]
    if mem_sec:
        parts.append(mem_sec)
    parts.append(instruction)
    return "\n\n".join(parts)
