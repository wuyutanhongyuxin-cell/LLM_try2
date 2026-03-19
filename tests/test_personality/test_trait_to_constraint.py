from __future__ import annotations

"""OCEAN -> 交易硬约束映射公式边界值测试（最关键的测试文件）。"""

import pytest

from src.personality.ocean_model import PRESET_PROFILES, OceanProfile
from src.personality.trait_to_constraint import TradingConstraints, ocean_to_constraints

# ── 全局配置 ──

GLOBAL_CFG: dict[str, list[str]] = {
    "major_assets": ["BTC-PERP", "ETH-PERP"],
    "all_assets": ["BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "DOGE-PERP"],
}


def _make_profile(**kwargs: int) -> OceanProfile:
    """快速创建测试用 OceanProfile，未指定维度默认 50。"""
    defaults = {"name": "test", "openness": 50, "conscientiousness": 50,
                "extraversion": 50, "agreeableness": 50, "neuroticism": 50}
    defaults.update(kwargs)
    return OceanProfile(**defaults)


def _constraints(**kwargs: int) -> TradingConstraints:
    """用指定的 OCEAN 参数生成 TradingConstraints。"""
    return ocean_to_constraints(_make_profile(**kwargs), GLOBAL_CFG)


# ── 神经质(N)相关：仓位、止损、回撤 ────────────────────

class TestNeuroticism:
    """N 越高 -> 仓位越小、止损越紧、回撤容忍越低。"""

    def test_n0_max_position(self) -> None:
        c = _constraints(neuroticism=0)
        assert c.max_position_pct == 30.0

    def test_n100_max_position(self) -> None:
        c = _constraints(neuroticism=100)
        assert c.max_position_pct == 5.0

    def test_n50_max_position(self) -> None:
        c = _constraints(neuroticism=50)
        # 5 + 50 * 0.25 = 17.5
        assert c.max_position_pct == 17.5

    def test_n0_stop_loss(self) -> None:
        c = _constraints(neuroticism=0)
        assert c.stop_loss_pct == 15.0

    def test_n100_stop_loss(self) -> None:
        c = _constraints(neuroticism=100)
        assert c.stop_loss_pct == 1.0

    def test_n50_stop_loss(self) -> None:
        c = _constraints(neuroticism=50)
        # 1 + 50 * 0.14 = 8.0
        assert c.stop_loss_pct == 8.0

    def test_n0_max_drawdown(self) -> None:
        c = _constraints(neuroticism=0)
        assert c.max_drawdown_pct == 20.0

    def test_n100_max_drawdown(self) -> None:
        c = _constraints(neuroticism=100)
        assert c.max_drawdown_pct == 2.0

    def test_n50_max_drawdown(self) -> None:
        c = _constraints(neuroticism=50)
        # 2 + 50 * 0.18 = 11.0
        assert c.max_drawdown_pct == 11.0


# ── 神经质(N)相关：再平衡间隔 ─────────────────────────

class TestRebalanceInterval:
    """N > 70 -> 300s, N > 40 -> 3600s, N <= 40 -> 86400s。"""

    def test_n71_rebalance_300(self) -> None:
        c = _constraints(neuroticism=71)
        assert c.rebalance_interval_seconds == 300

    def test_n100_rebalance_300(self) -> None:
        c = _constraints(neuroticism=100)
        assert c.rebalance_interval_seconds == 300

    def test_n70_rebalance_3600(self) -> None:
        """N=70 不满足 > 70，落入 > 40 区间。"""
        c = _constraints(neuroticism=70)
        assert c.rebalance_interval_seconds == 3600

    def test_n41_rebalance_3600(self) -> None:
        c = _constraints(neuroticism=41)
        assert c.rebalance_interval_seconds == 3600

    def test_n40_rebalance_86400(self) -> None:
        """N=40 不满足 > 40，落入 <= 40 区间。"""
        c = _constraints(neuroticism=40)
        assert c.rebalance_interval_seconds == 86400

    def test_n0_rebalance_86400(self) -> None:
        c = _constraints(neuroticism=0)
        assert c.rebalance_interval_seconds == 86400


# ── 开放性(O)相关：持仓数、允许资产 ─────────────────────

class TestOpenness:
    """O 高 -> 更多并发持仓，更广资产范围。"""

    def test_o0_max_concurrent(self) -> None:
        c = _constraints(openness=0)
        # 1 + 0 // 20 = 1
        assert c.max_concurrent_positions == 1

    def test_o100_max_concurrent(self) -> None:
        c = _constraints(openness=100)
        # 1 + 100 // 20 = 6, clip(6, 1, 6) = 6
        assert c.max_concurrent_positions == 6

    def test_o60_allowed_assets_major(self) -> None:
        """O=60 不满足 > 60，只能交易主流币。"""
        c = _constraints(openness=60)
        assert c.allowed_assets == ["BTC-PERP", "ETH-PERP"]

    def test_o61_allowed_assets_all(self) -> None:
        """O=61 满足 > 60，可交易全部币种。"""
        c = _constraints(openness=61)
        assert c.allowed_assets == GLOBAL_CFG["all_assets"]

    def test_o0_allowed_assets_major(self) -> None:
        c = _constraints(openness=0)
        assert c.allowed_assets == ["BTC-PERP", "ETH-PERP"]

    def test_o100_allowed_assets_all(self) -> None:
        c = _constraints(openness=100)
        assert c.allowed_assets == GLOBAL_CFG["all_assets"]


# ── 外向性(E)相关：情绪数据、动量权重 ───────────────────

class TestExtraversion:
    """E 高 -> 使用情绪数据，动量权重大。"""

    def test_e0_no_sentiment(self) -> None:
        c = _constraints(extraversion=0)
        assert c.use_sentiment_data is False

    def test_e50_no_sentiment(self) -> None:
        """E=50 不满足 > 50。"""
        c = _constraints(extraversion=50)
        assert c.use_sentiment_data is False

    def test_e51_use_sentiment(self) -> None:
        c = _constraints(extraversion=51)
        assert c.use_sentiment_data is True

    def test_e0_momentum_weight(self) -> None:
        c = _constraints(extraversion=0)
        assert c.momentum_weight == 0.0

    def test_e100_momentum_weight(self) -> None:
        c = _constraints(extraversion=100)
        assert c.momentum_weight == 1.0

    def test_e50_momentum_weight(self) -> None:
        c = _constraints(extraversion=50)
        assert c.momentum_weight == 0.5


# ── 宜人性(A)相关：逆向权重 ──────────────────────────────

class TestAgreeableness:
    """A 低 -> 逆向权重高。"""

    def test_a0_contrarian_weight(self) -> None:
        c = _constraints(agreeableness=0)
        assert c.contrarian_weight == 1.0

    def test_a100_contrarian_weight(self) -> None:
        c = _constraints(agreeableness=100)
        assert c.contrarian_weight == 0.0

    def test_a50_contrarian_weight(self) -> None:
        c = _constraints(agreeableness=50)
        assert c.contrarian_weight == 0.5


# ── 尽责性(C)相关：止损要求、信心阈值 ───────────────────

class TestConscientiousness:
    """C 高 -> 强制止损，信心阈值高。"""

    def test_c0_no_require_stop_loss(self) -> None:
        c = _constraints(conscientiousness=0)
        assert c.require_stop_loss is False

    def test_c50_no_require_stop_loss(self) -> None:
        """C=50 不满足 > 50。"""
        c = _constraints(conscientiousness=50)
        assert c.require_stop_loss is False

    def test_c51_require_stop_loss(self) -> None:
        c = _constraints(conscientiousness=51)
        assert c.require_stop_loss is True

    def test_c100_require_stop_loss(self) -> None:
        c = _constraints(conscientiousness=100)
        assert c.require_stop_loss is True

    def test_c0_min_confidence(self) -> None:
        c = _constraints(conscientiousness=0)
        # clip(0 / 100 * 0.8, 0.2, 0.8) = clip(0.0, 0.2, 0.8) = 0.2
        assert c.min_confidence_threshold == 0.2

    def test_c100_min_confidence(self) -> None:
        c = _constraints(conscientiousness=100)
        # clip(100 / 100 * 0.8, 0.2, 0.8) = clip(0.8, 0.2, 0.8) = 0.8
        assert c.min_confidence_threshold == 0.8

    def test_c50_min_confidence(self) -> None:
        c = _constraints(conscientiousness=50)
        # clip(50 / 100 * 0.8, 0.2, 0.8) = clip(0.4, 0.2, 0.8) = 0.4
        assert c.min_confidence_threshold == 0.4


# ── 对所有 32 个预定义原型运行映射，确保无异常 ───────────

class TestAllPresetsMapping:
    """确保全部预定义原型都能正确映射，不抛出任何异常。"""

    @pytest.mark.parametrize("name", list(PRESET_PROFILES.keys()))
    def test_preset_mapping_no_error(self, name: str) -> None:
        profile = PRESET_PROFILES[name]
        c = ocean_to_constraints(profile, GLOBAL_CFG)
        # 基本 sanity checks
        assert 5 <= c.max_position_pct <= 30
        assert 1 <= c.stop_loss_pct <= 15
        assert 2 <= c.max_drawdown_pct <= 20
        assert 1 <= c.max_concurrent_positions <= 6
        assert c.rebalance_interval_seconds in (300, 3600, 86400)
        assert 0.0 <= c.momentum_weight <= 1.0
        assert 0.0 <= c.contrarian_weight <= 1.0
        assert 0.2 <= c.min_confidence_threshold <= 0.8
        assert isinstance(c.allowed_assets, list)
        assert len(c.allowed_assets) >= 2
