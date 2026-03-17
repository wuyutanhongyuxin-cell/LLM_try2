from __future__ import annotations

"""ExecutionStrategy 测试 — 与 TestValidateSignal 对应，验证行为一致。"""

from datetime import datetime, timezone

import pytest

from src.execution.signal import Action
from src.execution.strategy import RuleBasedStrategy
from src.market.data_feed import MarketSnapshot
from src.personality.ocean_model import OceanProfile
from src.personality.trait_to_constraint import ocean_to_constraints
from tests.conftest import GLOBAL_CONFIG


@pytest.fixture()
def snapshot() -> MarketSnapshot:
    """默认行情快照。"""
    return MarketSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        asset="BTC-PERP", price=67200.0,
        price_24h_change_pct=1.5, volume_24h=5000000.0,
        high_24h=68000.0, low_24h=66000.0,
    )


def _make_strategy(
    openness: int = 50, conscientiousness: int = 50,
    extraversion: int = 50, agreeableness: int = 50, neuroticism: int = 50,
) -> tuple[RuleBasedStrategy, "TradingConstraints"]:
    """构建测试用 RuleBasedStrategy + constraints。"""
    profile = OceanProfile(
        name="测试型", openness=openness, conscientiousness=conscientiousness,
        extraversion=extraversion, agreeableness=agreeableness, neuroticism=neuroticism,
    )
    constraints = ocean_to_constraints(profile, GLOBAL_CONFIG)
    strategy = RuleBasedStrategy(
        agent_id="test_agent", agent_name="测试型",
        profile_dump=profile.model_dump(exclude={"name"}),
        prompt_hash="abc123", llm_model="test-model",
    )
    return strategy, constraints


def _valid_data(**overrides: object) -> dict:
    """生成合法的 LLM 解析后 dict。"""
    data = {
        "action": "BUY", "asset": "BTC-PERP", "size_pct": 10.0,
        "entry_price": 67200.0, "stop_loss_price": 64000.0,
        "take_profit_price": 72000.0, "confidence": 0.75,
        "reasoning": "bullish", "personality_influence": "O",
    }
    data.update(overrides)
    return data


class TestRuleBasedStrategy:
    """规则执行策略测试。"""

    def test_valid_signal_returns_trade_signal(self, snapshot: MarketSnapshot) -> None:
        """合法信号正常返回 TradeSignal。"""
        strategy, constraints = _make_strategy()
        signal = strategy.process_signal(_valid_data(), snapshot, constraints, 10000.0)
        assert signal is not None
        assert signal.action == Action.BUY

    def test_invalid_action_returns_none(self, snapshot: MarketSnapshot) -> None:
        """非法 action -> None。"""
        strategy, constraints = _make_strategy()
        signal = strategy.process_signal(_valid_data(action="INVALID"), snapshot, constraints, 10000.0)
        assert signal is None

    def test_disallowed_asset_returns_none(self, snapshot: MarketSnapshot) -> None:
        """低 O 不允许非主流资产。"""
        strategy, constraints = _make_strategy(openness=30)
        signal = strategy.process_signal(_valid_data(asset="SOL-PERP"), snapshot, constraints, 10000.0)
        assert signal is None

    def test_size_pct_clipped(self, snapshot: MarketSnapshot) -> None:
        """size_pct 超限被 clip。"""
        strategy, constraints = _make_strategy(neuroticism=100)  # max_position_pct=5
        signal = strategy.process_signal(_valid_data(size_pct=50.0), snapshot, constraints, 10000.0)
        assert signal is not None
        assert signal.size_pct == 5.0

    def test_confidence_below_threshold_returns_none(self, snapshot: MarketSnapshot) -> None:
        """信心不足 -> None。"""
        strategy, constraints = _make_strategy(conscientiousness=100)  # min_confidence=0.8
        signal = strategy.process_signal(_valid_data(confidence=0.5), snapshot, constraints, 10000.0)
        assert signal is None

    def test_require_stop_loss_but_missing(self, snapshot: MarketSnapshot) -> None:
        """缺止损且要求止损 -> None。"""
        strategy, constraints = _make_strategy(conscientiousness=80)
        signal = strategy.process_signal(_valid_data(stop_loss_price=None), snapshot, constraints, 10000.0)
        assert signal is None
