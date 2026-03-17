from __future__ import annotations

"""多采样投票决策模块。

多次调用 LLM 取多数票，提高决策一致性。
Self-Consistency 论文：1→3 次采样捕获约 80% 的一致性增益。
"""

from collections import Counter
from loguru import logger


def vote_on_actions(
    parsed_signals: list[dict],
    consensus_threshold: float = 0.6,
) -> dict | None:
    """对多个 LLM 解析结果进行 action 投票。

    规则：
    1. 按 action 分组计票
    2. 多数 action 占比 < consensus_threshold → 返回 None（无共识）
    3. 取多数 action 中 confidence 最高的信号作为最终结果

    Args:
        parsed_signals: 多次 LLM 调用解析后的 dict 列表
        consensus_threshold: 多数票占比阈值（默认 0.6）

    Returns:
        选中的信号 dict，或 None（无共识时默认 HOLD）
    """
    if not parsed_signals:
        return None
    # 统计 action 票数
    actions = [str(s.get("action", "HOLD")).upper() for s in parsed_signals]
    counts = Counter(actions)
    total = len(actions)
    winner, winner_count = counts.most_common(1)[0]
    ratio = winner_count / total
    logger.debug(f"多采样投票: {dict(counts)}, 胜出={winner} ({ratio:.0%})")
    if ratio < consensus_threshold:
        logger.info(f"多采样无共识: {dict(counts)}, 阈值={consensus_threshold}")
        return None
    # 从胜出 action 的信号中选 confidence 最高的
    candidates = [s for s in parsed_signals if str(s.get("action", "")).upper() == winner]
    best = max(candidates, key=lambda s: float(s.get("confidence", 0)))
    return best
