from __future__ import annotations

"""信号聚合器测试：independent 和 voting 两种模式。"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.aggregator import SignalAggregator
from src.execution.signal import Action, TradeSignal


def _make_signal(
    agent_id: str = "agent_a",
    action: Action = Action.BUY,
    asset: str = "BTC-PERP",
    confidence: float = 0.8,
    size_pct: float = 10.0,
    timestamp: datetime | None = None,
) -> TradeSignal:
    """快速构建测试用 TradeSignal。"""
    return TradeSignal(
        agent_id=agent_id,
        agent_name="测试型",
        timestamp=timestamp or datetime.now(tz=timezone.utc),
        action=action,
        asset=asset,
        size_pct=size_pct,
        entry_price=67200.0,
        stop_loss_price=64000.0,
        take_profit_price=72000.0,
        confidence=confidence,
        reasoning="test",
        personality_influence="test",
        ocean_profile={"openness": 50, "conscientiousness": 50,
                       "extraversion": 50, "agreeableness": 50, "neuroticism": 50},
    )


# ── Independent 模式 ────────────────────────────────────

class TestIndependentMode:
    """独立模式：信号直接转发给 paper_trader 执行。"""

    async def test_independent_returns_signal(self) -> None:
        """独立模式 handle_signal 返回信号本身。"""
        agg = SignalAggregator(mode="independent")
        signal = _make_signal()
        result = await agg.handle_signal(signal)
        assert result is not None
        assert result.agent_id == "agent_a"

    async def test_independent_calls_paper_trader(self) -> None:
        """独立模式调用 paper_trader.execute_signal。"""
        mock_pt = MagicMock()
        agg = SignalAggregator(mode="independent", paper_trader=mock_pt)
        signal = _make_signal()
        await agg.handle_signal(signal)
        mock_pt.execute_signal.assert_called_once_with(signal)

    async def test_independent_no_paper_trader(self) -> None:
        """paper_trader 为 None 时不崩溃。"""
        agg = SignalAggregator(mode="independent", paper_trader=None)
        signal = _make_signal()
        result = await agg.handle_signal(signal)
        assert result is not None


# ── Voting 模式 ─────────────────────────────────────────

class TestVotingMode:
    """投票模式：收集信号后加权聚合。"""

    async def test_voting_collects_within_window(self) -> None:
        """窗口内只收集不执行，返回 None。"""
        agg = SignalAggregator(mode="voting", signal_window_seconds=300)
        signal = _make_signal()
        result = await agg.handle_signal(signal)
        assert result is None  # 窗口未到期
        assert len(agg._pending_signals) == 1

    async def test_voting_aggregates_after_window(self) -> None:
        """窗口到期后聚合产出结果。"""
        agg = SignalAggregator(mode="voting", signal_window_seconds=1)
        # 发出第一个信号，时间戳在过去（超过窗口）
        old_time = datetime.now(tz=timezone.utc) - timedelta(seconds=10)
        signal1 = _make_signal(agent_id="a1", action=Action.BUY,
                               confidence=0.9, timestamp=old_time)
        # 第一个信号进入缓存
        await agg.handle_signal(signal1)
        # 第二个信号触发窗口到期检查
        signal2 = _make_signal(agent_id="a2", action=Action.BUY, confidence=0.7)
        result = await agg.handle_signal(signal2)
        assert result is not None
        assert result.action == Action.BUY
        assert result.agent_id == "aggregated"

    async def test_voting_hold_majority_returns_none(self) -> None:
        """多数投 HOLD 时返回 None。"""
        agg = SignalAggregator(mode="voting", signal_window_seconds=1)
        old_time = datetime.now(tz=timezone.utc) - timedelta(seconds=10)
        # 3 个 Agent 都投 HOLD
        for i in range(3):
            s = _make_signal(agent_id=f"agent_{i}", action=Action.HOLD,
                             confidence=0.9, timestamp=old_time if i == 0 else None)
            result = await agg.handle_signal(s)
        # 最后一个触发聚合，HOLD 多数 -> 不执行
        assert result is None

    async def test_voting_weighted_by_confidence_and_sharpe(self) -> None:
        """加权投票中 confidence x sharpe 影响结果。"""
        agg = SignalAggregator(mode="voting", signal_window_seconds=1)
        # 给 agent_a 很高的 sharpe
        agg.update_agent_sharpe("agent_a", 2.0)
        agg.update_agent_sharpe("agent_b", 0.1)
        old_time = datetime.now(tz=timezone.utc) - timedelta(seconds=10)
        # agent_a: BUY confidence=0.9, sharpe=2.0 -> weight=1.8
        s1 = _make_signal(agent_id="agent_a", action=Action.BUY,
                          confidence=0.9, size_pct=20.0, timestamp=old_time)
        await agg.handle_signal(s1)
        # agent_b: SELL confidence=0.5, sharpe=0.1 -> weight=0.05
        s2 = _make_signal(agent_id="agent_b", action=Action.SELL,
                          confidence=0.5, size_pct=5.0)
        result = await agg.handle_signal(s2)
        # BUY 的权重远大于 SELL，所以结果应为 BUY
        assert result is not None
        assert result.action == Action.BUY

    async def test_voting_clears_pending_after_aggregate(self) -> None:
        """聚合后清空 pending 缓存。"""
        agg = SignalAggregator(mode="voting", signal_window_seconds=1)
        old_time = datetime.now(tz=timezone.utc) - timedelta(seconds=10)
        s1 = _make_signal(agent_id="a1", action=Action.BUY, timestamp=old_time)
        await agg.handle_signal(s1)
        s2 = _make_signal(agent_id="a2", action=Action.BUY)
        await agg.handle_signal(s2)
        assert len(agg._pending_signals) == 0
