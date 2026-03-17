from __future__ import annotations

"""Prompt 生成器测试。"""

import pytest

from src.personality.ocean_model import OceanProfile
from src.personality.prompt_generator import (
    generate_decision_prompt,
    generate_system_prompt,
)
from src.personality.trait_to_constraint import TradingConstraints, ocean_to_constraints

from tests.conftest import CME_GLOBAL_CONFIG, GLOBAL_CONFIG


@pytest.fixture()
def profile_and_constraints() -> tuple[OceanProfile, TradingConstraints]:
    """返回一对 profile + constraints 用于测试。"""
    profile = OceanProfile(
        name="测试型", openness=70, conscientiousness=60,
        extraversion=40, agreeableness=30, neuroticism=50,
    )
    constraints = ocean_to_constraints(profile, GLOBAL_CONFIG)
    return profile, constraints


# ── System Prompt 测试 ──────────────────────────────────

class TestGenerateSystemPrompt:
    """验证 system prompt 结构和内容。"""

    def test_contains_role_section(
        self, profile_and_constraints: tuple[OceanProfile, TradingConstraints]
    ) -> None:
        """包含角色设定关键字。"""
        profile, constraints = profile_and_constraints
        prompt = generate_system_prompt(profile, constraints)
        assert "trader" in prompt and "personality" in prompt
        assert profile.name in prompt

    def test_contains_personality_section(
        self, profile_and_constraints: tuple[OceanProfile, TradingConstraints]
    ) -> None:
        """包含五维人格描述段落。"""
        profile, constraints = profile_and_constraints
        prompt = generate_system_prompt(profile, constraints)
        assert "Openness" in prompt
        assert "Conscientiousness" in prompt
        assert "Extraversion" in prompt
        assert "Agreeableness" in prompt
        assert "Neuroticism" in prompt

    def test_contains_constraints_section(
        self, profile_and_constraints: tuple[OceanProfile, TradingConstraints]
    ) -> None:
        """包含硬约束段落。"""
        profile, constraints = profile_and_constraints
        prompt = generate_system_prompt(profile, constraints)
        assert "HARD CONSTRAINTS" in prompt
        assert "max_position_pct" in prompt or "仓位" in prompt

    def test_contains_output_format(
        self, profile_and_constraints: tuple[OceanProfile, TradingConstraints]
    ) -> None:
        """包含 JSON 输出格式说明。"""
        profile, constraints = profile_and_constraints
        prompt = generate_system_prompt(profile, constraints)
        assert "Output Format" in prompt
        assert "JSON" in prompt

    def test_contains_rules_section(
        self, profile_and_constraints: tuple[OceanProfile, TradingConstraints]
    ) -> None:
        """包含禁止事项。"""
        profile, constraints = profile_and_constraints
        prompt = generate_system_prompt(profile, constraints)
        assert "Do NOT" in prompt

    def test_prompt_is_english(
        self, profile_and_constraints: tuple[OceanProfile, TradingConstraints]
    ) -> None:
        """Prompt 主体是英文（除了人格名称可能含中文）。"""
        profile, constraints = profile_and_constraints
        prompt = generate_system_prompt(profile, constraints)
        # 去掉人格名称后，英文字符应占主导
        prompt_no_name = prompt.replace(profile.name, "")
        ascii_chars = sum(1 for ch in prompt_no_name if ord(ch) < 128)
        assert ascii_chars / len(prompt_no_name) > 0.9


# ── Decision Prompt 测试 ────────────────────────────────

class TestGenerateDecisionPrompt:
    """验证 decision prompt 包含行情和持仓信息。"""

    def test_contains_market_data(self) -> None:
        """包含行情数据。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 2.5, "volume": 5000000.0}
        prompt = generate_decision_prompt(market, [], "", 10000.0)
        assert "BTC-PERP" in prompt
        assert "67,200.00" in prompt

    def test_contains_position_info(self) -> None:
        """持仓非空时包含持仓信息。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 0.0, "volume": 0.0}
        positions = [{"asset": "BTC-PERP", "size": 0.1,
                      "entry_price": 65000.0, "unrealized_pnl": 220.0}]
        prompt = generate_decision_prompt(market, positions, "", 10000.0)
        assert "Current Positions" in prompt
        assert "BTC-PERP" in prompt

    def test_no_positions_message(self) -> None:
        """无持仓时提示 No open positions。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 0.0, "volume": 0.0}
        prompt = generate_decision_prompt(market, [], "", 10000.0)
        assert "No open positions" in prompt

    def test_contains_portfolio_value(self) -> None:
        """包含资产总值。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 0.0, "volume": 0.0}
        prompt = generate_decision_prompt(market, [], "", 25000.0)
        assert "25,000.00" in prompt

    def test_contains_memory_context(self) -> None:
        """传入记忆上下文时应包含在 prompt 中。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 0.0, "volume": 0.0}
        ctx = "Recent trades: BUY BTC PnL=+100"
        prompt = generate_decision_prompt(market, [], ctx, 10000.0)
        assert "Memory" in prompt
        assert ctx in prompt

    def test_empty_memory_context_omitted(self) -> None:
        """空记忆上下文时不包含 Memory 段落。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 0.0, "volume": 0.0}
        prompt = generate_decision_prompt(market, [], "", 10000.0)
        assert "Memory" not in prompt

    def test_decision_prompt_is_english(self) -> None:
        """Decision prompt 也是英文。"""
        market = {"asset": "BTC-PERP", "price": 67200.0,
                  "change_24h": 1.0, "volume": 1000.0}
        prompt = generate_decision_prompt(market, [], "", 10000.0)
        ascii_chars = sum(1 for ch in prompt if ord(ch) < 128)
        assert ascii_chars / len(prompt) > 0.9


# ── CME 市场类型 Prompt 测试 ─────────────────────────────

class TestCMESystemPrompt:
    """验证 CME 市场类型生成的 system prompt 包含正确的市场上下文。"""

    def test_cme_role_mentions_futures(self) -> None:
        """CME 模式下角色包含 CME futures 关键字。"""
        profile = OceanProfile(
            name="测试型", openness=70, conscientiousness=60,
            extraversion=40, agreeableness=30, neuroticism=50,
        )
        constraints = ocean_to_constraints(profile, CME_GLOBAL_CONFIG)
        prompt = generate_system_prompt(profile, constraints, market_type="cme")
        assert "CME futures trader" in prompt

    def test_cme_asset_example(self) -> None:
        """CME 模式下资产示例是 ES 而非 BTC-PERP。"""
        profile = OceanProfile(
            name="测试型", openness=70, conscientiousness=60,
            extraversion=40, agreeableness=30, neuroticism=50,
        )
        constraints = ocean_to_constraints(profile, CME_GLOBAL_CONFIG)
        prompt = generate_system_prompt(profile, constraints, market_type="cme")
        assert '"ES"' in prompt
        assert "BTC-PERP" not in prompt

    def test_cme_openness_description(self) -> None:
        """CME 模式下高开放性描述与 crypto 不同。"""
        profile = OceanProfile(
            name="测试型", openness=80, conscientiousness=50,
            extraversion=50, agreeableness=50, neuroticism=50,
        )
        constraints = ocean_to_constraints(profile, CME_GLOBAL_CONFIG)
        prompt = generate_system_prompt(profile, constraints, market_type="cme")
        assert "energy" in prompt or "metals" in prompt or "diverse futures" in prompt

    def test_cme_allowed_assets_are_cme_symbols(self) -> None:
        """CME 模式下约束中的 allowed_assets 是 CME 合约代码。"""
        profile = OceanProfile(
            name="测试型", openness=80, conscientiousness=50,
            extraversion=50, agreeableness=50, neuroticism=50,
        )
        constraints = ocean_to_constraints(profile, CME_GLOBAL_CONFIG)
        assert "ES" in constraints.allowed_assets
        assert "BTC-PERP" not in constraints.allowed_assets


class TestCMEDecisionPrompt:
    """验证 CME 市场的 decision prompt。"""

    def test_cme_market_data(self) -> None:
        """CME 资产行情正确嵌入 decision prompt。"""
        market = {"asset": "ES", "price": 5900.0,
                  "change_24h": 0.5, "volume": 1200000.0}
        prompt = generate_decision_prompt(market, [], "", 50000.0)
        assert "ES" in prompt
        assert "5,900.00" in prompt

    def test_cme_knowledge_graph_lookup(self) -> None:
        """CME 资产名不含 -PERP，知识图谱查找使用原名。"""
        market = {"asset": "GC", "price": 3000.0,
                  "change_24h": 1.2, "volume": 80000.0}
        prompt = generate_decision_prompt(market, [], "", 50000.0)
        assert "MARKET KNOWLEDGE" in prompt
