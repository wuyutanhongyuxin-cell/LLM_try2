from __future__ import annotations

"""交易成本模型测试：滑点、手续费、资金费率计算。"""

import pytest

from src.execution.cost_model import (
    CMECostConfig,
    CostConfig,
    CostResult,
    calculate_cme_entry_cost,
    calculate_cme_exit_cost,
    calculate_entry_cost,
    calculate_exit_cost,
    calculate_funding_cost,
)


@pytest.fixture()
def default_config() -> CostConfig:
    """默认成本配置。"""
    return CostConfig()


class TestEntryCost:
    """开仓成本测试。"""

    def test_long_entry_slippage_increases_price(self, default_config: CostConfig) -> None:
        """LONG 开仓滑点应抬高成交价。"""
        result = calculate_entry_cost(
            price=10000.0, notional=10000.0, side="LONG", config=default_config
        )
        assert result.effective_price > 10000.0

    def test_short_entry_slippage_decreases_price(self, default_config: CostConfig) -> None:
        """SHORT 开仓滑点应压低成交价。"""
        result = calculate_entry_cost(
            price=10000.0, notional=10000.0, side="SHORT", config=default_config
        )
        assert result.effective_price < 10000.0

    def test_zero_slippage_effective_equals_original(self) -> None:
        """slippage_bps=0 时 effective_price 应等于原始价格。"""
        config = CostConfig(slippage_bps=0.0)
        result = calculate_entry_cost(
            price=50000.0, notional=10000.0, side="LONG", config=config
        )
        assert result.effective_price == 50000.0

    def test_fee_calculation(self) -> None:
        """notional=10000, taker_fee_rate=0.0004 -> fee=4.0。"""
        config = CostConfig(taker_fee_rate=0.0004, slippage_bps=0.0)
        result = calculate_entry_cost(
            price=50000.0, notional=10000.0, side="LONG", config=config
        )
        assert result.fee_cost == pytest.approx(4.0)


class TestExitCost:
    """平仓成本测试。"""

    def test_exit_cost_opposite_direction(self, default_config: CostConfig) -> None:
        """LONG 平仓应用 SHORT 方向滑点（成交价低于原价）。"""
        result = calculate_exit_cost(
            price=10000.0, notional=10000.0, side="LONG", config=default_config
        )
        # LONG 平仓 = SHORT 方向，滑点压低成交价
        assert result.effective_price < 10000.0


class TestFundingCost:
    """资金费率成本测试。"""

    def test_funding_cost_one_period(self, default_config: CostConfig) -> None:
        """8 小时 = 1 个 funding period。"""
        cost = calculate_funding_cost(
            notional=10000.0, holding_hours=8.0, config=default_config
        )
        expected = 10000.0 * default_config.funding_rate_8h * 1.0
        assert cost == pytest.approx(expected)

    def test_funding_cost_multiple_periods(self, default_config: CostConfig) -> None:
        """24 小时 = 3 个 funding periods。"""
        cost = calculate_funding_cost(
            notional=10000.0, holding_hours=24.0, config=default_config
        )
        expected = 10000.0 * default_config.funding_rate_8h * 3.0
        assert cost == pytest.approx(expected)


class TestCostsDisabled:
    """成本计算关闭时所有成本为 0。"""

    def test_costs_disabled(self) -> None:
        """enable_costs=False 时所有成本应为 0。"""
        config = CostConfig(enable_costs=False)
        entry = calculate_entry_cost(
            price=50000.0, notional=10000.0, side="LONG", config=config
        )
        assert entry.slippage_cost == 0.0
        assert entry.fee_cost == 0.0
        assert entry.total_cost == 0.0
        assert entry.effective_price == 50000.0

        funding = calculate_funding_cost(
            notional=10000.0, holding_hours=24.0, config=config
        )
        assert funding == 0.0


# ── CME 成本模型测试 ─────────────────────────────────────

class TestCMEEntryCost:
    """CME 期货开仓成本测试。"""

    def test_cme_long_slippage_increases_price(self) -> None:
        """CME LONG 开仓滑点应抬高成交价。"""
        config = CMECostConfig()
        result = calculate_cme_entry_cost(
            price=5900.0, contracts=1, multiplier=50.0,
            side="LONG", config=config,
        )
        assert result.effective_price > 5900.0

    def test_cme_short_slippage_decreases_price(self) -> None:
        """CME SHORT 开仓滑点应压低成交价。"""
        config = CMECostConfig()
        result = calculate_cme_entry_cost(
            price=5900.0, contracts=1, multiplier=50.0,
            side="SHORT", config=config,
        )
        assert result.effective_price < 5900.0

    def test_cme_commission_per_contract(self) -> None:
        """CME 佣金 = commission_per_contract * contracts。"""
        config = CMECostConfig(commission_per_contract=1.25, slippage_bps=0.0)
        result = calculate_cme_entry_cost(
            price=5900.0, contracts=5, multiplier=50.0,
            side="LONG", config=config,
        )
        assert result.fee_cost == pytest.approx(6.25)

    def test_cme_costs_disabled(self) -> None:
        """enable_costs=False 时 CME 成本为 0。"""
        config = CMECostConfig(enable_costs=False)
        result = calculate_cme_entry_cost(
            price=5900.0, contracts=2, multiplier=50.0,
            side="LONG", config=config,
        )
        assert result.total_cost == 0.0
        assert result.effective_price == 5900.0


class TestCMEExitCost:
    """CME 期货平仓成本测试。"""

    def test_cme_exit_reverses_direction(self) -> None:
        """LONG 平仓用 SHORT 方向计算。"""
        config = CMECostConfig()
        result = calculate_cme_exit_cost(
            price=5900.0, contracts=1, multiplier=50.0,
            side="LONG", config=config,
        )
        assert result.effective_price < 5900.0
