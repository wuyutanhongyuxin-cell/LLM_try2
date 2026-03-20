"""L4 永久长期记忆 — 投票写入 + 投票淘汰。

智慧提取：每 500 笔交易，10 轮 LLM 独立生成智慧，2/3 以上一致才写入。
记忆淘汰：每 1000 笔交易，10 轮 LLM 投票淘汰过时经验，2/3 以上票才删除。

文件结构：
  data/memory/{agent_id}/
    archive.jsonl   — 归档反思（可被投票淘汰）
    wisdom.md       — 投票通过的交易智慧摘要
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from litellm import acompletion

from src.agent.memory_pruner import apply_prune, vote_prune_entries
from src.personality.ocean_model import OceanProfile

_MEMORY_DIR = Path("data/memory")
_WISDOM_VOTE_ROUNDS = 10      # 智慧提取投票轮数
_WISDOM_THRESHOLD = 2 / 3     # 智慧写入阈值（≥7/10 一致）
_MIN_TRADES_FOR_WISDOM = 500  # 智慧提取最低交易数
_MIN_TRADES_FOR_PRUNE = 1000  # 淘汰检查最低交易数
_MIN_ENTRIES_FOR_PRUNE = 10   # 归档低于此数量不触发淘汰


class LongTermMemory:
    """永久长期记忆管理器。"""

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._dir = _MEMORY_DIR / agent_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._archive_path = self._dir / "archive.jsonl"
        self._wisdom_path = self._dir / "wisdom.md"

    def archive_reflection(self, reflection: str) -> None:
        """将反思原文追加到归档。"""
        entry = {"timestamp": datetime.now(tz=timezone.utc).isoformat(), "reflection": reflection}
        with open(self._archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"[{self._agent_id}] L4 归档反思 (共{self.get_archive_count()}条)")

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

    # ── 智慧提取（每 500 笔，10 轮投票） ──────────────

    async def extract_wisdom(
        self, profile: OceanProfile, llm_config: dict,
    ) -> bool:
        """10 轮 LLM 独立生成智慧，提取共识写入 wisdom.md。"""
        entries = self.get_all_archived()
        if not entries:
            return False
        prompt = _build_compress_prompt(profile, entries, self.get_wisdom())
        messages = [{"role": "user", "content": prompt}]
        # 10 轮独立生成
        candidates: list[str] = []
        for r in range(1, _WISDOM_VOTE_ROUNDS + 1):
            try:
                resp = await acompletion(
                    model=llm_config.get("model", "claude-sonnet-4-20250514"),
                    messages=messages, temperature=0.3, max_tokens=2048,
                )
                text: str = resp.choices[0].message.content.strip()  # type: ignore
                candidates.append(text)
                logger.debug(f"[{self._agent_id}] 智慧提取第{r}轮完成")
            except Exception as exc:
                logger.error(f"[{self._agent_id}] 智慧提取第{r}轮失败: {exc}")
        if not candidates:
            return False
        # 投票：选出出现最多的结构（按核心段落相似度）
        best = _select_consensus(candidates, _WISDOM_THRESHOLD)
        if best is None:
            logger.warning(f"[{self._agent_id}] 智慧提取未达共识，跳过写入")
            return False
        self._wisdom_path.write_text(best, encoding="utf-8")
        logger.info(f"[{self._agent_id}] L4 智慧已更新 (基于{len(entries)}条反思, {len(candidates)}轮投票)")
        return True

    # ── 淘汰检查（每 1000 笔） ────────────────────────

    async def prune_outdated(
        self, profile: OceanProfile, llm_config: dict,
        recent_trades: list[dict] | None = None,
    ) -> bool:
        """投票淘汰过时经验，淘汰后重新提取智慧。"""
        entries = self.get_all_archived()
        if len(entries) < _MIN_ENTRIES_FOR_PRUNE:
            return False
        logger.info(f"[{self._agent_id}] 触发记忆淘汰投票 (归档{len(entries)}条)")
        prune_ids = await vote_prune_entries(entries, profile, llm_config, recent_trades)
        if prune_ids:
            entries = apply_prune(entries, prune_ids)
            self._rewrite_archive(entries)
            # 淘汰后重新提取智慧
            return await self.extract_wisdom(profile, llm_config)
        return False

    def _rewrite_archive(self, entries: list[dict]) -> None:
        """淘汰后重写归档文件。"""
        with open(self._archive_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        logger.info(f"[{self._agent_id}] 归档已重写 (剩余{len(entries)}条)")


def _select_consensus(candidates: list[str], threshold: float) -> str | None:
    """从多轮生成结果中选出共识版本。

    策略：提取每个候选的核心段落标题，按结构相似度分组，
    超过 threshold 比例的最大组中选最长的版本。
    """
    # 提取结构指纹：##标题行集合
    def fingerprint(text: str) -> frozenset[str]:
        return frozenset(
            line.strip().lower() for line in text.split("\n")
            if line.strip().startswith("##")
        )
    fps = [fingerprint(c) for c in candidates]
    # 按指纹分组
    groups: dict[frozenset[str], list[int]] = {}
    for i, fp in enumerate(fps):
        matched = False
        for key in groups:
            # 允许 80% 标题重叠算同一组
            overlap = len(fp & key) / max(len(fp | key), 1)
            if overlap >= 0.8:
                groups[key].append(i)
                matched = True
                break
        if not matched:
            groups[fp] = [i]
    # 找最大组
    biggest = max(groups.values(), key=len)
    ratio = len(biggest) / len(candidates)
    if ratio < threshold:
        return None
    # 选最大组中最长的版本（信息最丰富）
    best_idx = max(biggest, key=lambda i: len(candidates[i]))
    return candidates[best_idx]


def _build_compress_prompt(
    profile: OceanProfile, entries: list[dict], current_wisdom: str,
) -> str:
    """构建智慧压缩 prompt。"""
    traits = (f"O={profile.openness} C={profile.conscientiousness} "
              f"E={profile.extraversion} A={profile.agreeableness} N={profile.neuroticism}")
    recent = entries[-100:]
    refs = "\n".join(f"[{e['timestamp'][:10]}] {e['reflection']}" for e in recent)
    existing = f"\n\nPrevious wisdom:\n{current_wisdom}" if current_wisdom else ""
    return (
        f"You are '{profile.name}', a crypto trader ({traits}).\n\n"
        f"Trade reflections:\n{refs}\n{existing}\n\n"
        "Synthesize into a TRADING WISDOM document. Structure:\n"
        "## Core Trading Lessons\n## Strategy Evolution\n"
        "## Known Blind Spots\n## Market Regime Notes\n"
        "## Personality-Specific Rules\n\n"
        "Be concise but comprehensive. This is your trading bible."
    )
