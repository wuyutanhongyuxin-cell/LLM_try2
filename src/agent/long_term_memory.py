"""L4 永久长期记忆 — 持久化交易智慧 + 投票淘汰过时经验。

L3 反思最多 20 条会被淘汰，L4 将所有反思归档到本地文件，
定期用 LLM 压缩成「交易智慧」摘要，并通过多轮投票淘汰错误经验。

文件结构：
  data/memory/{agent_id}/
    archive.jsonl   — 归档反思（可被投票淘汰）
    wisdom.md       — LLM 压缩后的交易智慧摘要（定期更新）
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from litellm import acompletion

from src.agent.memory_pruner import apply_prune, vote_prune_entries
from src.personality.ocean_model import OceanProfile

_MEMORY_DIR = Path("data/memory")
_MAX_RETRIES = 2
_MIN_ENTRIES_FOR_PRUNE = 10   # 归档低于此数量不触发淘汰
_MIN_TRADES_FOR_PRUNE = 500   # 总交易数低于此不触发淘汰


class LongTermMemory:
    """永久长期记忆管理器。"""

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._dir = _MEMORY_DIR / agent_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._archive_path = self._dir / "archive.jsonl"
        self._wisdom_path = self._dir / "wisdom.md"

    def archive_reflection(self, reflection: str) -> None:
        """将反思原文追加到永久归档（append-only，永不删除）。"""
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "reflection": reflection,
        }
        with open(self._archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count = self.get_archive_count()
        logger.info(f"[{self._agent_id}] L4 归档反思 (共{count}条)")

    def get_archive_count(self) -> int:
        """获取归档反思总数。"""
        if not self._archive_path.exists():
            return 0
        with open(self._archive_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def get_all_archived(self) -> list[dict]:
        """读取全部归档反思。"""
        if not self._archive_path.exists():
            return []
        entries: list[dict] = []
        with open(self._archive_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def get_wisdom(self) -> str:
        """读取当前交易智慧摘要。"""
        if not self._wisdom_path.exists():
            return ""
        return self._wisdom_path.read_text(encoding="utf-8").strip()

    async def review_and_compress(
        self, profile: OceanProfile, llm_config: dict,
        recent_trades: list[dict] | None = None,
        total_trades: int = 0,
    ) -> bool:
        """先投票淘汰过时经验，再压缩交易智慧。

        淘汰条件（全部满足才触发）：
        - 归档反思 >= 10 条
        - 总交易数 >= 500 笔（确保有足够数据支撑判断）
        """
        entries = self.get_all_archived()
        if not entries:
            return False
        # 步骤1：投票淘汰（条件严格：归档>=10 且 总交易>=500）
        if len(entries) >= _MIN_ENTRIES_FOR_PRUNE and total_trades >= _MIN_TRADES_FOR_PRUNE:
            logger.info(
                f"[{self._agent_id}] 触发记忆淘汰投票 "
                f"(归档{len(entries)}条, 交易{total_trades}笔)"
            )
            prune_ids = await vote_prune_entries(
                entries, profile, llm_config, recent_trades,
            )
            if prune_ids:
                entries = apply_prune(entries, prune_ids)
                self._rewrite_archive(entries)
        # 步骤2：压缩智慧
        return await self._compress(profile, llm_config, entries)

    async def _compress(
        self, profile: OceanProfile, llm_config: dict,
        entries: list[dict],
    ) -> bool:
        """用 LLM 将归档反思压缩成交易智慧摘要。"""
        current_wisdom = self.get_wisdom()
        prompt = _build_compress_prompt(profile, entries, current_wisdom)
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await acompletion(
                    model=llm_config.get("model", "claude-sonnet-4-20250514"),
                    messages=messages, temperature=0.2, max_tokens=2048,
                )
                wisdom: str = resp.choices[0].message.content  # type: ignore
                self._wisdom_path.write_text(wisdom.strip(), encoding="utf-8")
                logger.info(
                    f"[{self._agent_id}] L4 交易智慧已更新 "
                    f"(基于{len(entries)}条反思)"
                )
                return True
            except Exception as exc:
                logger.error(
                    f"[{self._agent_id}] L4 压缩失败 "
                    f"(第{attempt+1}次): {exc}"
                )
        return False

    def _rewrite_archive(self, entries: list[dict]) -> None:
        """淘汰后重写归档文件。"""
        with open(self._archive_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        logger.info(f"[{self._agent_id}] 归档已重写 (剩余{len(entries)}条)")


def _build_compress_prompt(
    profile: OceanProfile,
    entries: list[dict],
    current_wisdom: str,
) -> str:
    """构建智慧压缩 prompt。"""
    traits = (
        f"O={profile.openness} C={profile.conscientiousness} "
        f"E={profile.extraversion} A={profile.agreeableness} "
        f"N={profile.neuroticism}"
    )
    # 取最近 100 条防止 prompt 过长
    recent = entries[-100:]
    reflections_text = "\n".join(
        f"[{e['timestamp'][:10]}] {e['reflection']}" for e in recent
    )
    existing = f"\n\nPrevious wisdom summary:\n{current_wisdom}" if current_wisdom else ""
    return (
        f"You are '{profile.name}', a crypto trader ({traits}).\n\n"
        f"Below are ALL your trade reflections to date:\n"
        f"{reflections_text}\n"
        f"{existing}\n\n"
        "Synthesize these into a COMPREHENSIVE TRADING WISDOM document. "
        "This is your permanent memory — it will be loaded into every "
        "future trading decision.\n\n"
        "Structure your output as:\n"
        "## Core Trading Lessons\n"
        "- Key patterns you've identified\n\n"
        "## Strategy Evolution\n"
        "- How your approach has changed over time\n\n"
        "## Known Blind Spots\n"
        "- Recurring mistakes to watch for\n\n"
        "## Market Regime Notes\n"
        "- What works in different market conditions\n\n"
        "## Personality-Specific Rules\n"
        "- Rules derived from your OCEAN profile's strengths/weaknesses\n\n"
        "Be concise but comprehensive. This is your trading bible."
    )
