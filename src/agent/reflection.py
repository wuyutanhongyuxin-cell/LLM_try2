"""Agent 交易后反思模块。每完成 10 笔交易触发一次。"""

from __future__ import annotations

import json

from loguru import logger
from litellm import acompletion

from src.personality.ocean_model import OceanProfile

# 反思要求 LLM 输出的 JSON schema 说明（嵌入 prompt）
_OUTPUT_SCHEMA = """\
{
    "lessons": ["lesson 1", "lesson 2", "lesson 3"],
    "personality_observation": "How your personality affected these trades",
    "adjustment_suggestion": "What you would do differently next time",
    "emotional_state": "Your current emotional state as a trader",
    "summary": "One-paragraph summary of the reflection"
}"""

_MAX_RETRIES = 2


def _build_reflection_prompt(profile: OceanProfile, recent_trades: list[dict]) -> str:
    """构建反思 prompt（全英文）。"""
    traits = (
        f"Openness={profile.openness}, Conscientiousness={profile.conscientiousness}, "
        f"Extraversion={profile.extraversion}, Agreeableness={profile.agreeableness}, "
        f"Neuroticism={profile.neuroticism}"
    )
    trades_text = json.dumps(recent_trades, indent=2, default=str)
    return (
        f"You are a cryptocurrency trader named '{profile.name}' "
        f"with the following Big Five personality: {traits}.\n\n"
        f"Below are your most recent trades:\n{trades_text}\n\n"
        "Reflect on these trades from your personality's perspective. "
        "Consider how your traits influenced your decisions, what went well, "
        "and what could be improved.\n\n"
        f"Respond with ONLY a valid JSON object matching this schema:\n{_OUTPUT_SCHEMA}"
    )


async def generate_reflection(
    agent_name: str,
    profile: OceanProfile,
    recent_trades: list[dict],
    llm_config: dict,
) -> dict | None:
    """生成交易反思，LLM 调用失败或 JSON 解析失败返回 None。"""
    prompt = _build_reflection_prompt(profile, recent_trades)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(_MAX_RETRIES):
        try:
            resp = await acompletion(
                model=llm_config.get("model", "claude-sonnet-4-20250514"),
                messages=messages,
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 1024),
            )
            raw: str = resp.choices[0].message.content  # type: ignore[union-attr]
            result: dict = json.loads(raw)
            logger.info("[{}] 反思生成成功", agent_name)
            return result
        except json.JSONDecodeError as exc:
            logger.warning("[{}] 反思 JSON 解析失败 (第{}次): {}", agent_name, attempt + 1, exc)
        except Exception as exc:
            logger.error("[{}] 反思 LLM 调用失败 (第{}次): {}", agent_name, attempt + 1, exc)

    logger.error("[{}] 反思生成最终失败，已重试{}次", agent_name, _MAX_RETRIES)
    return None


# ── 元反思（二阶反思）──────────────────────────────

_META_OUTPUT_SCHEMA = """\
{
    "meta_lessons": ["跨多次反思识别的模式1", "模式2"],
    "strategy_evolution": "策略如何演化的总结",
    "blind_spots": "反复出现但未改善的问题",
    "meta_summary": "一段话的元反思总结"
}"""


def _build_meta_reflection_prompt(
    profile: OceanProfile,
    recent_reflections: list[str],
) -> str:
    """构建元反思 prompt：对最近 N 条反思进行二阶反思。"""
    traits = (
        f"Openness={profile.openness}, Conscientiousness={profile.conscientiousness}, "
        f"Extraversion={profile.extraversion}, Agreeableness={profile.agreeableness}, "
        f"Neuroticism={profile.neuroticism}"
    )
    refs_text = "\n".join(f"  [{i+1}] {r}" for i, r in enumerate(recent_reflections))
    return (
        f"You are a cryptocurrency trader named '{profile.name}' "
        f"with the following Big Five personality: {traits}.\n\n"
        f"Below are your most recent trade reflections:\n{refs_text}\n\n"
        "Perform a META-REFLECTION: analyze these reflections themselves. "
        "Identify recurring patterns, how your strategy has evolved, "
        "and blind spots you keep missing.\n\n"
        f"Respond with ONLY a valid JSON object matching this schema:\n{_META_OUTPUT_SCHEMA}"
    )


async def generate_meta_reflection(
    agent_name: str,
    profile: OceanProfile,
    recent_reflections: list[str],
    llm_config: dict,
) -> dict | None:
    """生成元反思（二阶反思），调用逻辑与 generate_reflection 类似。"""
    prompt = _build_meta_reflection_prompt(profile, recent_reflections)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(_MAX_RETRIES):
        try:
            resp = await acompletion(
                model=llm_config.get("model", "claude-sonnet-4-20250514"),
                messages=messages,
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 1024),
            )
            raw: str = resp.choices[0].message.content  # type: ignore[union-attr]
            result: dict = json.loads(raw)
            logger.info("[{}] 元反思生成成功", agent_name)
            return result
        except json.JSONDecodeError as exc:
            logger.warning("[{}] 元反思 JSON 解析失败 (第{}次): {}", agent_name, attempt + 1, exc)
        except Exception as exc:
            logger.error("[{}] 元反思 LLM 调用失败 (第{}次): {}", agent_name, attempt + 1, exc)

    logger.error("[{}] 元反思生成最终失败", agent_name)
    return None
