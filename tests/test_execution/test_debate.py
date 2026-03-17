from __future__ import annotations

"""Bull/Bear 辩论模块测试。"""

from datetime import datetime, timezone

from src.execution.debate import _build_debate_prompt, apply_debate_result
from src.execution.signal import Action, TradeSignal


def _make_signal(
    action: Action = Action.BUY,
    confidence: float = 0.7,
    reasoning: str = "bullish pattern",
    size_pct: float = 10.0,
    agent_name: str = "测试型",
) -> TradeSignal:
    """快速构建测试用 TradeSignal。"""
    return TradeSignal(
        agent_id="test", agent_name=agent_name,
        timestamp=datetime.now(tz=timezone.utc),
        action=action, asset="BTC-PERP", size_pct=size_pct,
        entry_price=67200.0, stop_loss_price=64000.0,
        take_profit_price=72000.0, confidence=confidence,
        reasoning=reasoning, personality_influence="test",
        ocean_profile={"openness": 50, "conscientiousness": 50,
                       "extraversion": 50, "agreeableness": 50, "neuroticism": 50},
    )


class TestBuildDebatePrompt:
    """辩论 prompt 构建测试。"""

    def test_groups_buy_sell_hold(self) -> None:
        """BUY/SELL/HOLD 的 reasoning 应被正确分组。"""
        signals = [
            _make_signal(action=Action.BUY, reasoning="going up"),
            _make_signal(action=Action.SELL, reasoning="going down"),
            _make_signal(action=Action.HOLD, reasoning="stay flat"),
        ]
        prompt = _build_debate_prompt(signals, "BTC-PERP")
        assert "going up" in prompt
        assert "going down" in prompt
        assert "stay flat" in prompt
        assert "Bull Arguments" in prompt
        assert "Bear Arguments" in prompt


class TestApplyDebateResult:
    """辩论结果应用测试。"""

    def test_bull_boosts_buy_confidence(self) -> None:
        """BULL 结果应提升 BUY 信号 confidence。"""
        signals = [_make_signal(action=Action.BUY, confidence=0.6)]
        result = {"dominant_view": "BULL", "confidence_adjustment": 0.2}
        adjusted = apply_debate_result(signals, result)
        assert adjusted[0].confidence > 0.6

    def test_bull_reduces_sell_confidence(self) -> None:
        """BULL 结果应降低 SELL 信号 confidence。"""
        signals = [_make_signal(action=Action.SELL, confidence=0.6)]
        result = {"dominant_view": "BULL", "confidence_adjustment": 0.2}
        adjusted = apply_debate_result(signals, result)
        assert adjusted[0].confidence < 0.6

    def test_confidence_clipped_to_0_1(self) -> None:
        """confidence 被 clip 到 [0.0, 1.0] 范围。"""
        signals = [_make_signal(action=Action.BUY, confidence=0.95)]
        result = {"dominant_view": "BULL", "confidence_adjustment": 0.3}
        adjusted = apply_debate_result(signals, result)
        assert adjusted[0].confidence <= 1.0

    def test_does_not_modify_action_or_size(self) -> None:
        """辩论结果不修改 action 和 size_pct。"""
        signals = [_make_signal(action=Action.BUY, size_pct=15.0)]
        result = {"dominant_view": "BULL", "confidence_adjustment": 0.1}
        adjusted = apply_debate_result(signals, result)
        assert adjusted[0].action == Action.BUY
        assert adjusted[0].size_pct == 15.0

    def test_empty_signals_no_error(self) -> None:
        """空信号列表不报错。"""
        result = {"dominant_view": "BULL", "confidence_adjustment": 0.2}
        adjusted = apply_debate_result([], result)
        assert adjusted == []

    def test_zero_adjustment_no_change(self) -> None:
        """confidence_adjustment 为 0 时信号不变。"""
        signals = [_make_signal(action=Action.BUY, confidence=0.7)]
        result = {"dominant_view": "BULL", "confidence_adjustment": 0.0}
        adjusted = apply_debate_result(signals, result)
        assert adjusted[0].confidence == 0.7
