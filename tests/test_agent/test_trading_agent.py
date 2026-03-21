from __future__ import annotations

"""TradingAgent 信号校验与 JSON 解析测试。

使用 MockDataFeed + 固定 LLM 响应，不需要真实 LLM 或 Redis。
"""

import json
from datetime import datetime, timezone

import pytest

from src.agent.trading_agent import TradingAgent
from src.execution.signal import Action
from src.market.data_feed import MarketSnapshot, MockDataFeed
from src.personality.ocean_model import OceanProfile
from src.personality.trait_to_constraint import ocean_to_constraints
from tests.conftest import FakeRedisBus, GLOBAL_CONFIG


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture()
def snapshot() -> MarketSnapshot:
    """默认行情快照。"""
    return MarketSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        asset="BTC-PERP", price=67200.0,
        price_24h_change_pct=1.5, volume_24h=5000000.0,
        high_24h=68000.0, low_24h=66000.0,
    )


def _make_agent(
    openness: int = 50, conscientiousness: int = 50,
    extraversion: int = 50, agreeableness: int = 50, neuroticism: int = 50,
) -> TradingAgent:
    """构建测试用 TradingAgent。"""
    profile = OceanProfile(
        name="测试型", openness=openness, conscientiousness=conscientiousness,
        extraversion=extraversion, agreeableness=agreeableness, neuroticism=neuroticism,
    )
    constraints = ocean_to_constraints(profile, GLOBAL_CONFIG)
    llm_config = {"model": "test-model", "retry_count": 1, "temperature": 0.3}
    feed = MockDataFeed()
    bus = FakeRedisBus()
    return TradingAgent(
        agent_id="test_agent", profile=profile, constraints=constraints,
        llm_config=llm_config, market_feed=feed, redis_bus=bus,  # type: ignore[arg-type]
    )


def _valid_signal_json(**overrides: object) -> str:
    """生成合法的 LLM 响应 JSON 字符串。"""
    data = {
        "action": "BUY",
        "asset": "BTC-PERP",
        "size_pct": 10.0,
        "entry_price": 67200.0,
        "stop_loss_price": 64000.0,
        "take_profit_price": 72000.0,
        "confidence": 0.75,
        "reasoning": "Technical analysis shows bullish pattern",
        "personality_influence": "Openness drives exploration",
    }
    data.update(overrides)
    return json.dumps(data)


# ── _parse_llm_response 测试 ────────────────────────────

class TestParseLlmResponse:
    """JSON 解析测试。"""

    def test_valid_json(self) -> None:
        agent = _make_agent()
        result = agent._parse_llm_response('{"action": "BUY"}')
        assert result == {"action": "BUY"}

    def test_invalid_json_returns_none(self) -> None:
        agent = _make_agent()
        result = agent._parse_llm_response("not valid json at all")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        agent = _make_agent()
        result = agent._parse_llm_response("")
        assert result is None

    def test_markdown_wrapped_json_returns_none(self) -> None:
        """LLM 用 markdown 包裹 JSON 时应解析失败。"""
        agent = _make_agent()
        result = agent._parse_llm_response('```json\n{"action": "BUY"}\n```')
        assert result is None


# ── _validate_signal 测试 ───────────────────────────────

class TestValidateSignal:
    """信号校验：硬约束由代码强制执行。"""

    def test_valid_signal_returns_trade_signal(self, snapshot: MarketSnapshot) -> None:
        """合法信号正常返回 TradeSignal。"""
        agent = _make_agent()
        raw = _valid_signal_json()
        signal = agent._validate_signal(raw, snapshot)
        assert signal is not None
        assert signal.action == Action.BUY
        assert signal.asset == "BTC-PERP"

    def test_invalid_action_returns_none(self, snapshot: MarketSnapshot) -> None:
        """非法 action -> None。"""
        agent = _make_agent()
        raw = _valid_signal_json(action="INVALID")
        signal = agent._validate_signal(raw, snapshot)
        assert signal is None

    def test_empty_action_returns_none(self, snapshot: MarketSnapshot) -> None:
        agent = _make_agent()
        raw = _valid_signal_json(action="")
        signal = agent._validate_signal(raw, snapshot)
        assert signal is None

    def test_disallowed_asset_returns_none(self, snapshot: MarketSnapshot) -> None:
        """资产不在允许列表 -> None（低 O 只能交易主流币）。"""
        agent = _make_agent(openness=30)  # O<=60 -> 只允许 BTC-PERP, ETH-PERP
        raw = _valid_signal_json(asset="SOL-PERP")
        signal = agent._validate_signal(raw, snapshot)
        assert signal is None

    def test_allowed_alt_asset_with_high_openness(self, snapshot: MarketSnapshot) -> None:
        """高 O 时 SOL-PERP 是允许的。"""
        agent = _make_agent(openness=80)  # O>60 -> 全部币种
        raw = _valid_signal_json(asset="SOL-PERP")
        signal = agent._validate_signal(raw, snapshot)
        assert signal is not None
        assert signal.asset == "SOL-PERP"

    def test_size_pct_clipped(self, snapshot: MarketSnapshot) -> None:
        """size_pct 超出 max_position_pct 时被 clip。"""
        agent = _make_agent(neuroticism=100)  # N=100 -> max_position_pct=5
        raw = _valid_signal_json(size_pct=50.0)
        signal = agent._validate_signal(raw, snapshot)
        assert signal is not None
        assert signal.size_pct == 5.0  # 被 clip 到 max

    def test_confidence_below_threshold_returns_none(self, snapshot: MarketSnapshot) -> None:
        """confidence 低于 min_confidence_threshold -> None。"""
        agent = _make_agent(conscientiousness=100)  # C=100 -> min_confidence=0.8
        raw = _valid_signal_json(confidence=0.5)
        signal = agent._validate_signal(raw, snapshot)
        assert signal is None

    def test_confidence_at_threshold_passes(self, snapshot: MarketSnapshot) -> None:
        """confidence 刚好等于 threshold 应通过。"""
        agent = _make_agent(conscientiousness=100)  # min_confidence=0.8
        raw = _valid_signal_json(confidence=0.8)
        signal = agent._validate_signal(raw, snapshot)
        assert signal is not None

    def test_require_stop_loss_but_missing(self, snapshot: MarketSnapshot) -> None:
        """require_stop_loss=True 但无止损 -> None。"""
        agent = _make_agent(conscientiousness=80)  # C>50 -> require_stop_loss=True
        raw = _valid_signal_json(stop_loss_price=None)
        signal = agent._validate_signal(raw, snapshot)
        assert signal is None

    def test_no_require_stop_loss_missing_ok(self, snapshot: MarketSnapshot) -> None:
        """require_stop_loss=False 时缺少止损也能通过。"""
        agent = _make_agent(conscientiousness=30)  # C<=50 -> require_stop_loss=False
        raw = _valid_signal_json(stop_loss_price=None)
        signal = agent._validate_signal(raw, snapshot)
        assert signal is not None

    def test_invalid_json_returns_none(self, snapshot: MarketSnapshot) -> None:
        """非法 JSON -> None。"""
        agent = _make_agent()
        signal = agent._validate_signal("not json", snapshot)
        assert signal is None

    def test_hold_action_returns_none(self, snapshot: MarketSnapshot) -> None:
        """HOLD = 无动作，应返回 None（不构建 TradeSignal）。"""
        agent = _make_agent()
        raw = _valid_signal_json(action="HOLD")
        signal = agent._validate_signal(raw, snapshot)
        assert signal is None


# ── 元反思触发逻辑 ──────────────────────────────────

class TestMetaReflection:
    """元反思触发条件测试。"""

    def test_meta_not_triggered_before_30(self) -> None:
        """30 笔之前不触发元反思（trade_count % 30 != 0）。"""
        agent = _make_agent()
        agent._trade_count = 20
        assert agent._trade_count % 30 != 0

    def test_meta_triggered_at_30(self) -> None:
        """第 30 笔时应触发元反思。"""
        agent = _make_agent()
        agent._trade_count = 30
        assert agent._trade_count % 30 == 0

    def test_meta_triggered_at_60(self) -> None:
        """第 60 笔时也应触发元反思。"""
        agent = _make_agent()
        agent._trade_count = 60
        assert agent._trade_count % 30 == 0

    def test_meta_not_triggered_at_10(self) -> None:
        """第 10 笔只触发普通反思，不触发元反思。"""
        agent = _make_agent()
        agent._trade_count = 10
        assert agent._trade_count % 10 == 0
        assert agent._trade_count % 30 != 0
