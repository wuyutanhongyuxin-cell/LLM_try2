"""L4 永久长期记忆 — 类似 Claude Memory 的持久化交易智慧。

L3 反思最多 20 条会被淘汰，L4 将所有反思归档到本地文件，
并定期用 LLM 压缩成「交易智慧」摘要，永久保留、越跑越聪明。

文件结构：
  data/memory/{agent_id}/
    archive.jsonl   — 全部反思原文（追加写入，永不删除）
    wisdom.md       — LLM 压缩后的交易智慧摘要（定期更新）
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from litellm import acompletion

from src.personality.ocean_model import OceanProfile

_MEMORY_DIR = Path("data/memory")
_MAX_RETRIES = 2


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

    async def compress_wisdom(
        self, profile: OceanProfile, llm_config: dict,
    ) -> bool:
        """用 LLM 将全部归档反思压缩成交易智慧摘要，覆写 wisdom.md。"""
        entries = self.get_all_archived()
        if not entries:
            return False
        current_wisdom = self.get_wisdom()
        prompt = _build_compress_prompt(profile, entries, current_wisdom)
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await acompletion(
                    model=llm_config.get("model", "claude-sonnet-4-20250514"),
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
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
