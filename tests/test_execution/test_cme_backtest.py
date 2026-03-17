"""CME 期货回测端到端测试（不调用 LLM，验证成本路径正确）。"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.execution.cost_model import CMECostConfig, CostConfig
from src.execution.account import AgentAccount
from src.execution.paper_trader import PaperTrader
from src.execution.signal import Action, TradeSignal


def _make_cme_signal(
    agent_id: str = "test_cme",
    action: Action = Action.BUY,
    asset: str = "ES",
    size_pct: float = 10.0,
    entry_price: float = 5900.0,
    stop_loss_price: float = 5850.0,
    take_profit_price: float = 6000.0,
) -> TradeSignal:
    """构造 CME 测试信号。"""
    return TradeSignal(
        agent_id=agent_id, agent_name="CME测试",
        timestamp=datetime.now(tz=timezone.utc),
        action=action, asset=asset, size_pct=size_pct,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        confidence=0.8, reasoning="test",
        personality_influence="test",
        ocean_profile={"O": 50, "C": 50, "E": 50, "A": 50, "N": 50},
    )


class TestCMECostPath:
    """验证 CME 成本通过 account.py 正确执行。"""

    def test_cme_account_uses_commission(self) -> None:
        """CME 账户应使用 commission_per_contract 计算成本。"""
        cme_cfg = CMECostConfig(commission_per_contract=2.50, slippage_bps=0)
        acc = AgentAccount(
            "test", Decimal("100000"),
            cme_cost_config=cme_cfg,
            contract_multiplier=50.0,
        )
        signal = _make_cme_signal(size_pct=10.0, entry_price=5900.0)
        acc.execute_buy(signal, {"ES": 5900.0})
        # 佣金 > 0，所以 total_costs 应 > 0
        assert acc.total_costs > Decimal("0")

    def test_cme_account_no_funding_rate(self) -> None:
        """CMECostConfig 不应有 funding_rate_8h 字段。"""
        cme_cfg = CMECostConfig(commission_per_contract=1.25)
        assert not hasattr(cme_cfg, "funding_rate_8h")

    def test_default_registration_is_crypto(self) -> None:
        """默认注册（不传 CME 参数）仍是加密货币模式。"""
        pt = PaperTrader()
        pt.register_agent("crypto_agent", 10000.0)
        assert pt._accounts["crypto_agent"]._is_cme is False

    def test_cme_registration_sets_flag(self) -> None:
        """CME 注册应设置 _is_cme 标志。"""
        pt = PaperTrader()
        cme_cfg = CMECostConfig()
        pt.register_agent("cme_agent", 100000.0,
                          cme_cost_config=cme_cfg,
                          contract_multiplier=50.0)
        assert pt._accounts["cme_agent"]._is_cme is True

    def test_cme_cost_higher_than_zero_cost(self) -> None:
        """开启成本的 CME 账户应比关闭成本时总资产低。"""
        # 有成本
        cme_on = CMECostConfig(commission_per_contract=1.25, slippage_bps=2)
        acc_on = AgentAccount(
            "on", Decimal("100000"),
            cme_cost_config=cme_on, contract_multiplier=50.0)
        # 无成本
        cme_off = CMECostConfig(
            commission_per_contract=1.25, slippage_bps=2, enable_costs=False)
        acc_off = AgentAccount(
            "off", Decimal("100000"),
            cme_cost_config=cme_off, contract_multiplier=50.0)
        signal = _make_cme_signal()
        acc_on.execute_buy(signal, {"ES": 5900.0})
        acc_off.execute_buy(signal, {"ES": 5900.0})
        assert acc_on.total_costs > acc_off.total_costs


class TestCMEBuyExecute:
    """CME 买入执行测试。"""

    def test_buy_es_creates_position(self) -> None:
        """买入 ES 后应有持仓。"""
        cme_cfg = CMECostConfig(slippage_bps=1, commission_per_contract=1.25)
        pt = PaperTrader()
        pt.register_agent("test_cme", 100000.0,
                          cme_cost_config=cme_cfg,
                          contract_multiplier=50.0)
        pt._current_prices = {"ES": 5900.0}
        signal = _make_cme_signal(agent_id="test_cme")
        result = pt.execute_signal(signal)
        assert result is True
        assert len(pt._accounts["test_cme"].positions) == 1

    def test_buy_then_sell_records_trade(self) -> None:
        """买入后卖出应记录平仓交易。"""
        cme_cfg = CMECostConfig(slippage_bps=1, commission_per_contract=1.25)
        acc = AgentAccount(
            "test", Decimal("100000"),
            cme_cost_config=cme_cfg, contract_multiplier=50.0)
        buy_sig = _make_cme_signal(action=Action.BUY, entry_price=5900.0)
        acc.execute_buy(buy_sig, {"ES": 5900.0})
        assert len(acc.positions) == 1
        sell_sig = _make_cme_signal(action=Action.SELL, entry_price=5950.0)
        acc.execute_sell(sell_sig, {"ES": 5950.0})
        assert len(acc.positions) == 0
        assert len(acc.closed_trades) == 1

    def test_stop_loss_triggers_for_cme(self) -> None:
        """CME 持仓止损应正常触发。"""
        cme_cfg = CMECostConfig(slippage_bps=0, commission_per_contract=0.0)
        acc = AgentAccount(
            "test", Decimal("100000"),
            cme_cost_config=cme_cfg, contract_multiplier=50.0)
        buy_sig = _make_cme_signal(
            entry_price=5900.0, stop_loss_price=5850.0)
        acc.execute_buy(buy_sig, {"ES": 5900.0})
        # 价格跌到止损线
        events = acc.check_stop_loss_take_profit("ES", 5850.0)
        assert len(events) == 1
        assert events[0]["reason"] == "STOP_LOSS"
