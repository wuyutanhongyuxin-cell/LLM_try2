"""多 Agent 信号聚合器。

支持 independent（独立执行）和 voting（加权投票）两种模式。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.execution.debate import apply_debate_result, run_debate
from src.execution.signal import Action, TradeSignal

if TYPE_CHECKING:
    from src.execution.paper_trader import PaperTrader
    from src.integration.redis_bus import RedisBus


class SignalAggregator:
    """信号聚合器，订阅 agent_signals 并聚合。"""

    def __init__(self, mode: str, signal_window_seconds: int = 120,
                 paper_trader: PaperTrader | None = None,
                 redis_bus: RedisBus | None = None,
                 enable_debate: bool = False,
                 llm_config: dict | None = None) -> None:
        self._mode = mode
        self._window = signal_window_seconds
        self._paper_trader = paper_trader
        self._redis_bus = redis_bus
        self._enable_debate = enable_debate
        self._llm_config = llm_config or {}
        self._pending_signals: list[TradeSignal] = []  # 投票模式缓存
        self._agent_sharpes: dict[str, float] = {}     # agent_id → 历史 Sharpe
        logger.info("信号聚合器初始化 | 模式={} 辩论={}", mode, enable_debate)

    async def handle_signal(self, signal: TradeSignal) -> TradeSignal | None:
        """处理单个信号。独立模式直接执行，投票模式先缓存。"""
        if self._mode == "independent":
            await self._execute_independent(signal)
            return signal
        return await self._collect_for_voting(signal)

    def update_agent_sharpe(self, agent_id: str, sharpe: float) -> None:
        """更新 Agent 历史 Sharpe（供加权使用）。"""
        self._agent_sharpes[agent_id] = sharpe

    async def _execute_independent(self, signal: TradeSignal) -> None:
        """独立模式：直接转发信号给 paper_trader 执行。"""
        logger.info("独立模式 | 转发信号 agent={} action={}", signal.agent_id, signal.action.value)
        if self._paper_trader is not None:
            self._paper_trader.execute_signal(signal)

    async def _collect_for_voting(self, signal: TradeSignal) -> TradeSignal | None:
        """投票模式：收集信号，窗口到期后聚合。"""
        self._pending_signals.append(signal)
        # 以第一条信号的时间戳为窗口起点，用最新信号的时间戳判断是否到期
        first_ts = self._pending_signals[0].timestamp
        latest_ts = signal.timestamp
        if latest_ts - first_ts < timedelta(seconds=self._window):
            return None  # 窗口未到期
        signals = self._pending_signals.copy()
        self._pending_signals.clear()
        return await self._aggregate_votes(signals)

    async def _aggregate_votes(self, signals: list[TradeSignal]) -> TradeSignal | None:
        """加权投票聚合：confidence × sharpe 决定方向，仓位取加权平均。"""
        if not signals:
            return None
        # 按 asset 分组，取信号最多的资产
        asset_groups: dict[str, list[TradeSignal]] = {}
        for s in signals:
            asset_groups.setdefault(s.asset, []).append(s)
        target_asset = max(asset_groups, key=lambda a: len(asset_groups[a]))
        group = asset_groups[target_asset]
        # 辩论环节（仅 voting 模式且启用时）
        if self._enable_debate and self._llm_config:
            debate_result = await run_debate(group, target_asset, self._llm_config)
            if debate_result:
                group = apply_debate_result(group, debate_result)
                logger.info("辩论完成 | 主导观点={}", debate_result.get("dominant_view"))
        # 计算各方向加权得分
        scores: dict[Action, float] = {a: 0.0 for a in Action}
        w_sizes: dict[Action, float] = {a: 0.0 for a in Action}
        w_sums: dict[Action, float] = {a: 0.0 for a in Action}
        for s in group:
            w = s.confidence * max(self._agent_sharpes.get(s.agent_id, 0.1), 0.1)
            scores[s.action] += w
            w_sizes[s.action] += s.size_pct * w
            w_sums[s.action] += w
        # 多数派方向
        winner = max(scores, key=lambda a: scores[a])
        if winner == Action.HOLD:
            logger.info("投票聚合 | 结果=HOLD，不执行")
            return None
        avg_size = w_sizes[winner] / w_sums[winner] if w_sums[winner] > 0 else 0.0
        # 取该方向中 confidence 最高的信号作为模板
        best = max((s for s in group if s.action == winner), key=lambda s: s.confidence)
        aggregated = best.model_copy(update={
            "agent_id": "aggregated", "agent_name": "投票聚合",
            "size_pct": round(avg_size, 2),
            "reasoning": f"投票聚合({len(group)}个信号): {winner.value}",
            "timestamp": datetime.now(tz=timezone.utc),
        })
        logger.info("投票聚合 | 方向={} 仓位={:.2f}% 信号数={}", winner.value, avg_size, len(group))
        return aggregated
