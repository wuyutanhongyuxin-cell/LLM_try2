"""L4 记忆投票淘汰模块 — 多轮 LLM 投票清理过时/错误经验。

当归档反思积累到一定量后，用 3 轮 LLM 投票识别过时经验，
超过 2/3 票的条目被淘汰。防止错误经验污染后续决策。
"""

from __future__ import annotations

import json
from collections import Counter

from loguru import logger
from litellm import acompletion

from src.personality.ocean_model import OceanProfile

_VOTE_ROUNDS = 3          # 投票轮数
_PRUNE_THRESHOLD = 2 / 3  # 淘汰阈值：超过 2/3 票


async def vote_prune_entries(
    entries: list[dict],
    profile: OceanProfile,
    llm_config: dict,
    recent_trades: list[dict] | None = None,
) -> list[int]:
    """多轮投票识别应淘汰的归档条目索引。

    流程：
    1. 将全部归档反思 + 最近交易表现发给 LLM
    2. LLM 返回应淘汰的条目编号列表
    3. 重复 3 轮，超过 2/3 票的条目被淘汰

    Returns:
        应淘汰的条目索引列表
    """
    if len(entries) < 10:
        return []  # 太少不值得淘汰
    prompt = _build_prune_prompt(entries, profile, recent_trades)
    messages = [{"role": "user", "content": prompt}]
    # 3 轮独立投票
    all_votes: list[list[int]] = []
    for round_num in range(1, _VOTE_ROUNDS + 1):
        try:
            resp = await acompletion(
                model=llm_config.get("model", "claude-sonnet-4-20250514"),
                messages=messages,
                temperature=0.5,  # 稍高温度保证多样性
                max_tokens=512,
            )
            raw: str = resp.choices[0].message.content  # type: ignore
            indices = _parse_indices(raw, max_idx=len(entries) - 1)
            all_votes.append(indices)
            logger.debug(
                f"淘汰投票第{round_num}轮: 提名 {len(indices)} 条"
            )
        except Exception as exc:
            logger.error(f"淘汰投票第{round_num}轮失败: {exc}")
            all_votes.append([])
    # 统计票数，超过 2/3 的淘汰
    vote_counts: Counter[int] = Counter()
    for votes in all_votes:
        for idx in votes:
            vote_counts[idx] += 1
    threshold = int(_VOTE_ROUNDS * _PRUNE_THRESHOLD)  # 3 × 2/3 = 2
    pruned = [idx for idx, cnt in vote_counts.items() if cnt >= threshold]
    if pruned:
        logger.info(
            f"投票淘汰: {len(pruned)} 条经验被淘汰 "
            f"(共{len(entries)}条, 阈值≥{threshold}票)"
        )
    return sorted(pruned)


def apply_prune(entries: list[dict], prune_indices: list[int]) -> list[dict]:
    """从归档列表中移除被淘汰的条目。"""
    prune_set = set(prune_indices)
    return [e for i, e in enumerate(entries) if i not in prune_set]


def _parse_indices(raw: str, max_idx: int) -> list[int]:
    """从 LLM 响应中解析条目编号列表。"""
    # 尝试 JSON 数组解析
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [int(x) for x in data if 0 <= int(x) <= max_idx]
        if isinstance(data, dict) and "indices" in data:
            return [int(x) for x in data["indices"] if 0 <= int(x) <= max_idx]
    except (json.JSONDecodeError, ValueError):
        pass
    # 兜底：提取文本中的数字
    indices: list[int] = []
    for token in raw.replace(",", " ").replace("[", " ").replace("]", " ").split():
        try:
            n = int(token)
            if 0 <= n <= max_idx:
                indices.append(n)
        except ValueError:
            pass
    return indices


def _build_prune_prompt(
    entries: list[dict],
    profile: OceanProfile,
    recent_trades: list[dict] | None,
) -> str:
    """构建淘汰投票 prompt。"""
    # 编号列出全部归档反思
    lines: list[str] = []
    for i, e in enumerate(entries):
        ts = e.get("timestamp", "")[:10]
        ref = str(e.get("reflection", ""))[:200]
        lines.append(f"[{i}] ({ts}) {ref}")
    entries_text = "\n".join(lines)
    # 最近交易表现（如果有）
    perf_text = ""
    if recent_trades:
        perf_text = "\n\nRecent trade results:\n" + json.dumps(
            recent_trades[-10:], indent=1, default=str,
        )
    return (
        f"You are '{profile.name}', reviewing your trading memory archive.\n\n"
        f"Below are ALL archived reflections, each with an index number:\n"
        f"{entries_text}\n"
        f"{perf_text}\n\n"
        "Identify reflections that are NOW OUTDATED, PROVEN WRONG, "
        "or CONTRADICTED by more recent experience.\n\n"
        "Criteria for removal:\n"
        "- The lesson was based on a specific market regime that no longer applies\n"
        "- Later reflections explicitly contradict this one\n"
        "- The advice led to repeated losses in recent trades\n"
        "- The reflection is a duplicate of another one\n\n"
        "If ALL reflections are still valid, return an empty list: []\n"
        "Otherwise, return ONLY a JSON array of index numbers to remove.\n"
        "Example: [2, 5, 11]\n\n"
        "Respond with ONLY the JSON array, nothing else."
    )
