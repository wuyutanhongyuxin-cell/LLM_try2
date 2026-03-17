"""Bull/Bear 辩论模块（TradingAgents 启发）。

在 voting 聚合模式中，收集所有 Agent 的 reasoning 后，
调用一个独立的 "裁判 LLM" 做结构化辩论综合判断。

设计原则：
1. 不共享 Agent 记忆（只用信号中的公开 reasoning 字段）
2. 裁判 LLM 是独立调用，不是任何 Agent
3. 输出结构化 JSON，不是自由文本
4. 辩论结果仅作为 voting 的参考权重调整，不直接决定方向

参考：TradingAgents (arxiv 2412.20138) 的 Bull/Bear Researcher 设计
"""
from __future__ import annotations

import json

from litellm import acompletion
from loguru import logger

from src.execution.signal import Action, TradeSignal

# 裁判 LLM 输出 JSON 格式
_JUDGE_OUTPUT_SCHEMA: str = """{
  "dominant_view": "BULL | BEAR | NEUTRAL",
  "confidence_adjustment": "float between -0.3 and +0.3",
  "key_argument": "the strongest argument from either side",
  "risk_flag": "string or null"
}"""


def _build_debate_prompt(signals: list[TradeSignal], asset: str) -> str:
    """构建辩论 prompt，将 Agent reasoning 分为 Bull/Bear/Neutral 三组。"""
    bull_args: list[str] = []
    bear_args: list[str] = []
    neutral_args: list[str] = []
    for s in signals:
        entry = f"[{s.agent_name}] (conf={s.confidence:.2f}): {s.reasoning}"
        if s.action == Action.BUY:
            bull_args.append(entry)
        elif s.action == Action.SELL:
            bear_args.append(entry)
        else:
            neutral_args.append(entry)
    # 拼装 prompt
    parts: list[str] = [
        f"You are a neutral market analysis judge for {asset}.",
        "Evaluate the following arguments from multiple trading agents.",
        "",
        "## Bull Arguments (BUY):",
        ("\n".join(bull_args) if bull_args else "(none)"),
        "",
        "## Bear Arguments (SELL):",
        ("\n".join(bear_args) if bear_args else "(none)"),
        "",
        "## Neutral Arguments (HOLD):",
        ("\n".join(neutral_args) if neutral_args else "(none)"),
        "",
        "Based on the strength of arguments, respond with ONLY a JSON object:",
        _JUDGE_OUTPUT_SCHEMA,
    ]
    return "\n".join(parts)


async def run_debate(
    signals: list[TradeSignal],
    asset: str,
    llm_config: dict,
) -> dict | None:
    """执行辩论并返回裁判结果。

    Args:
        signals: 同一时间窗口内所有 Agent 的信号
        asset: 辩论的目标资产
        llm_config: LLM 配置（model, temperature 等）

    Returns:
        裁判判断 dict，或 None（LLM 调用失败时）
    """
    if not signals:
        return None
    prompt = _build_debate_prompt(signals, asset)
    try:
        resp = await acompletion(
            model=llm_config.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config.get("temperature", 0.2),
            max_tokens=512,
            timeout=llm_config.get("timeout_seconds", 30),
        )
        raw = resp.choices[0].message.content  # type: ignore[union-attr]
        return json.loads(raw)
    except Exception as exc:
        logger.error(f"辩论 LLM 调用失败: {exc}")
        return None


def apply_debate_result(
    signals: list[TradeSignal],
    debate_result: dict,
) -> list[TradeSignal]:
    """根据辩论结果调整信号的 confidence。

    规则：
    - dominant_view == "BULL" → BUY +adj, SELL -adj
    - dominant_view == "BEAR" → SELL +adj, BUY -adj
    - "NEUTRAL" → 不调整
    - confidence 被 clip 到 [0.0, 1.0]
    - 不修改 action、size_pct 等其他字段
    - 返回新的 TradeSignal 列表（model_copy）
    """
    view = debate_result.get("dominant_view", "NEUTRAL").upper()
    adj = float(debate_result.get("confidence_adjustment", 0.0))
    # clip adjustment 到合理范围
    adj = max(-0.3, min(0.3, adj))
    if view == "NEUTRAL" or adj == 0.0:
        return [s.model_copy() for s in signals]
    adjusted: list[TradeSignal] = []
    for s in signals:
        delta = 0.0
        if view == "BULL":
            delta = adj if s.action == Action.BUY else (-adj if s.action == Action.SELL else 0.0)
        elif view == "BEAR":
            delta = adj if s.action == Action.SELL else (-adj if s.action == Action.BUY else 0.0)
        new_conf = max(0.0, min(1.0, s.confidence + delta))
        adjusted.append(s.model_copy(update={"confidence": round(new_conf, 4)}))
    return adjusted
