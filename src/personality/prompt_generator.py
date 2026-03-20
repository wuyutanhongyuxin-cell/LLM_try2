from __future__ import annotations

"""Prompt 生成器：OCEAN 人格 → LLM Prompt。支持 crypto/cme 市场。"""

import hashlib

from src.personality.ocean_model import OceanProfile
from src.personality.prompt_constants import (
    ASSET_DESCRIPTIONS, ROLE_TEMPLATES, TRAIT_DESC_CRYPTO, TRAIT_DESCS,
)
from src.personality.trait_to_constraint import TradingConstraints
from src.utils.knowledge_graph import build_knowledge_context


def _build_personality_section(
    profile: OceanProfile, market_type: str = "crypto"
) -> str:
    """段落2：逐维度列出分数及交易行为含义。"""
    trait_desc = TRAIT_DESCS.get(market_type, TRAIT_DESC_CRYPTO)
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
    role_tmpl = ROLE_TEMPLATES.get(market_type, ROLE_TEMPLATES["crypto"])
    role = role_tmpl.format(name=profile.name)
    personality = _build_personality_section(profile, market_type)
    hard = _build_constraints_section(constraints)
    # 资产示例根据市场类型调整
    asset_example = '"BTC-PERP"' if market_type == "crypto" else '"ES"'
    # Fix 1: JSON 示例 + confidence 校准（DeepSeek 对完整示例响应最佳）
    json_example = (
        '{"action": "BUY", "asset": ' + asset_example + ', '
        '"size_pct": 15.0, "entry_price": 6650.00, '
        '"stop_loss_price": 6580.00, "take_profit_price": 6750.00, '
        '"confidence": 0.55, '
        '"reasoning": "RSI oversold at 28, MACD bullish crossover", '
        '"personality_influence": "Low neuroticism allows holding through volatility"}'
    )
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
        '- "personality_influence": string (which trait dominated this decision)\n\n'
        f"Example response:\n{json_example}"
    )
    # 行动指引：鼓励 LLM 主动决策，避免永远 HOLD
    action_guide = (
        "## Decision Guidelines\n"
        "- You are an ACTIVE trader, not a passive observer. "
        "Analyze the data and take positions when you see opportunity.\n"
        "- BUY = bullish signal, open or add to long position.\n"
        "- SELL = bearish signal, close existing position or express a short view. "
        "You SHOULD use SELL when you believe the price will drop, "
        "regardless of whether you currently hold a position.\n"
        "- HOLD should ONLY be used when the data is genuinely ambiguous "
        "or contradicts your trading style. Do NOT default to HOLD out of caution.\n"
        "- If technical indicators show a clear trend (RSI oversold/overbought, "
        "MACD crossover, price above/below SMA), you SHOULD act on it.\n"
        "- Your personality defines HOW you trade (aggressive vs conservative sizing, "
        "tight vs loose stops), not WHETHER you trade.\n"
        "- Confidence: 0.3 = low but tradeable, 0.5 = moderate, 0.7+ = high. "
        "0.4 is sufficient to act. confidence=0.0 is ONLY for HOLD."
    )
    rules = (
        "## Rules\n"
        "- Do NOT fabricate market data or prices.\n"
        "- Do NOT exceed any hard constraint listed above.\n"
        "- Do NOT output anything other than the JSON object.\n"
        "- Do NOT wrap the JSON in markdown code fences.\n"
        "You MUST respond with ONLY a valid JSON object."
    )
    prompt = "\n\n".join([role, personality, hard, output_fmt, action_guide, rules])
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
    max_positions: int = 6,
) -> str:
    """生成 Decision Prompt（英文），每次决策循环调用。

    Args:
        max_positions: 最大同时持仓数（用于显示持仓使用情况）
    """
    # 行情
    asset = market_data.get("asset", "UNKNOWN")
    price = market_data.get("price", 0)
    change = market_data.get("change_24h", 0)
    volume = market_data.get("volume", 0)
    market_lines = [
        "## Current Market Data",
        f"- Asset: {asset}",
    ]
    # Fix 6: 品种特化描述注入，帮助 LLM 理解非主流品种
    asset_desc = ASSET_DESCRIPTIONS.get(asset, "")
    if asset_desc:
        market_lines.append(f"- Asset Context: {asset_desc}")
    market_lines.extend([
        f"- Price: ${price:,.2f}",
        f"- 24h Change: {change:+.2f}%",
        f"- 24h Volume: ${volume:,.0f}",
    ])
    # 技术指标（如果提供）
    if "rsi_14" in market_data:
        rsi = market_data["rsi_14"]
        rsi_label = "OVERSOLD" if rsi < 30 else ("OVERBOUGHT" if rsi > 70 else "NEUTRAL")
        market_lines.append(f"- RSI(14): {rsi} [{rsi_label}]")
    if "sma_20" in market_data:
        market_lines.append(
            f"- SMA(20): ${market_data['sma_20']:,.2f} "
            f"(price is {market_data.get('price_vs_sma', 'N/A')} SMA)")
    if "macd_histogram" in market_data:
        market_lines.append(
            f"- MACD Histogram: {market_data['macd_histogram']:.4f} "
            f"[{market_data.get('macd_signal', 'N/A')}]")
    market_sec = "\n".join(market_lines)
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
    # Fix 5: 显示持仓占比，帮助 LLM 了解资金使用情况
    used = sum(p.get("size", 0) * p.get("entry_price", 0) for p in positions)
    available = portfolio_value - used
    avail_pct = (available / portfolio_value * 100) if portfolio_value > 0 else 0
    port_sec = (
        f"## Portfolio\n"
        f"- Total Value: ${portfolio_value:,.2f}\n"
        f"- Available Balance: ${available:,.2f} ({avail_pct:.1f}% of portfolio)\n"
        f"- Positions Used: {len(positions)}/{max_positions}"
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
