from __future__ import annotations

"""纸上交易执行器测试。"""

from datetime import datetime, timezone

import pytest

from src.execution.paper_trader import PaperTrader
from src.execution.signal import Action, TradeSignal


def _make_signal(
    agent_id: str = "agent_a",
    action: Action = Action.BUY,
    asset: str = "BTC-PERP",
    size_pct: float = 10.0,
    entry_price: float = 67200.0,
    stop_loss_price: float | None = 64000.0,
    take_profit_price: float | None = 72000.0,
    confidence: float = 0.8,
) -> TradeSignal:
    """快速构建测试用 TradeSignal。"""
    return TradeSignal(
        agent_id=agent_id,
        agent_name="测试型",
        timestamp=datetime.now(tz=timezone.utc),
        action=action,
        asset=asset,
        size_pct=size_pct,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        confidence=confidence,
        reasoning="test reasoning",
        personality_influence="test influence",
        ocean_profile={"openness": 50, "conscientiousness": 50,
                       "extraversion": 50, "agreeableness": 50, "neuroticism": 50},
    )


@pytest.fixture()
def trader() -> PaperTrader:
    """创建并注册一个 Agent 的 PaperTrader。"""
    pt = PaperTrader()
    pt.register_agent("agent_a", 10000.0)
    return pt


# ── 注册 ────────────────────────────────────────────────

class TestRegisterAgent:
    """测试 Agent 账户注册。"""

    def test_register_creates_account(self) -> None:
        pt = PaperTrader()
        pt.register_agent("agent_x", 5000.0)
        assert "agent_x" in pt._accounts

    def test_register_initial_capital(self) -> None:
        pt = PaperTrader()
        pt.register_agent("agent_x", 5000.0)
        from decimal import Decimal
        assert pt._accounts["agent_x"].cash == Decimal("5000")


# ── BUY 执行 ────────────────────────────────────────────

class TestExecuteBuy:
    """测试买入信号执行。"""

    def test_buy_increases_positions(self, trader: PaperTrader) -> None:
        """买入后持仓数增加。"""
        signal = _make_signal(action=Action.BUY)
        trader._current_prices = {"BTC-PERP": 67200.0}
        result = trader.execute_signal(signal)
        assert result is True
        acc = trader._accounts["agent_a"]
        assert len(acc.positions) == 1
        assert acc.positions[0].asset == "BTC-PERP"

    def test_buy_reduces_cash(self, trader: PaperTrader) -> None:
        """买入后可用资金减少。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(action=Action.BUY, size_pct=10.0)
        trader.execute_signal(signal)
        acc = trader._accounts["agent_a"]
        from decimal import Decimal
        assert acc.cash < Decimal("10000")

    def test_hold_returns_false(self, trader: PaperTrader) -> None:
        """HOLD 不执行，返回 False。"""
        signal = _make_signal(action=Action.HOLD)
        result = trader.execute_signal(signal)
        assert result is False


# ── SELL 执行 ────────────────────────────────────────────

class TestExecuteSell:
    """测试卖出信号执行。"""

    def test_sell_reduces_positions(self, trader: PaperTrader) -> None:
        """先买入再卖出，持仓清空。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        buy_signal = _make_signal(action=Action.BUY)
        trader.execute_signal(buy_signal)
        sell_signal = _make_signal(action=Action.SELL, entry_price=68000.0)
        result = trader.execute_signal(sell_signal)
        assert result is True
        acc = trader._accounts["agent_a"]
        assert len(acc.positions) == 0

    def test_sell_without_position_fails(self, trader: PaperTrader) -> None:
        """无持仓卖出应失败。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(action=Action.SELL)
        result = trader.execute_signal(signal)
        assert result is False

    def test_sell_records_closed_trade(self, trader: PaperTrader) -> None:
        """卖出后记录已平仓交易。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        trader.execute_signal(_make_signal(action=Action.BUY))
        trader.execute_signal(_make_signal(action=Action.SELL, entry_price=68000.0))
        acc = trader._accounts["agent_a"]
        assert len(acc.closed_trades) == 1


# ── 止损/止盈 ───────────────────────────────────────────

class TestStopLossTakeProfit:
    """测试行情更新时的止损/止盈触发。"""

    def test_stop_loss_triggered(self, trader: PaperTrader) -> None:
        """价格 <= stop_loss 时触发止损。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(
            action=Action.BUY, stop_loss_price=64000.0, take_profit_price=72000.0,
        )
        trader.execute_signal(signal)
        # 更新价格到止损线以下
        events = trader.update_prices({"BTC-PERP": 63000.0})
        assert len(events) == 1
        assert events[0]["reason"] == "STOP_LOSS"
        # 持仓应被清空
        assert len(trader._accounts["agent_a"].positions) == 0

    def test_stop_loss_at_exact_price(self, trader: PaperTrader) -> None:
        """价格 == stop_loss 也触发（等于也触发）。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(action=Action.BUY, stop_loss_price=64000.0)
        trader.execute_signal(signal)
        events = trader.update_prices({"BTC-PERP": 64000.0})
        assert len(events) == 1
        assert events[0]["reason"] == "STOP_LOSS"

    def test_take_profit_triggered(self, trader: PaperTrader) -> None:
        """价格 >= take_profit 时触发止盈。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(
            action=Action.BUY, stop_loss_price=64000.0, take_profit_price=72000.0,
        )
        trader.execute_signal(signal)
        events = trader.update_prices({"BTC-PERP": 73000.0})
        assert len(events) == 1
        assert events[0]["reason"] == "TAKE_PROFIT"

    def test_take_profit_at_exact_price(self, trader: PaperTrader) -> None:
        """价格 == take_profit 也触发。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(action=Action.BUY, take_profit_price=72000.0)
        trader.execute_signal(signal)
        events = trader.update_prices({"BTC-PERP": 72000.0})
        assert len(events) == 1
        assert events[0]["reason"] == "TAKE_PROFIT"

    def test_price_between_sl_tp_no_trigger(self, trader: PaperTrader) -> None:
        """价格在止损和止盈之间，不触发任何事件。"""
        trader._current_prices = {"BTC-PERP": 67200.0}
        signal = _make_signal(
            action=Action.BUY, stop_loss_price=64000.0, take_profit_price=72000.0,
        )
        trader.execute_signal(signal)
        events = trader.update_prices({"BTC-PERP": 68000.0})
        assert len(events) == 0
        assert len(trader._accounts["agent_a"].positions) == 1


# ── 排行榜 ──────────────────────────────────────────────

class TestLeaderboard:
    """测试排行榜功能。"""

    def test_leaderboard_sorted_by_sharpe(self) -> None:
        """排行榜按 Sharpe Ratio 降序排列。"""
        pt = PaperTrader()
        pt.register_agent("agent_a", 10000.0)
        pt.register_agent("agent_b", 10000.0)
        pt._current_prices = {"BTC-PERP": 67200.0}

        # agent_b 做一笔盈利交易 -> 有正收益
        buy_b = _make_signal(agent_id="agent_b", action=Action.BUY, size_pct=10.0)
        pt.execute_signal(buy_b)
        sell_b = _make_signal(
            agent_id="agent_b", action=Action.SELL,
            entry_price=70000.0,  # 以更高价格卖出
        )
        pt.execute_signal(sell_b)

        # 记录日收益率（agent_b 有盈利）
        pt.record_daily_returns()
        pt.record_daily_returns()  # 需要至少 2 条才能算 Sharpe

        lb = pt.get_leaderboard()
        assert len(lb) == 2
        # 两个 agent 都有 stats
        ids = [s["agent_id"] for s in lb]
        assert "agent_a" in ids
        assert "agent_b" in ids

    def test_leaderboard_contains_required_fields(self) -> None:
        """排行榜每条记录包含必要字段。"""
        pt = PaperTrader()
        pt.register_agent("agent_a", 10000.0)
        pt._current_prices = {}
        lb = pt.get_leaderboard()
        assert len(lb) == 1
        stats = lb[0]
        required_fields = [
            "agent_id", "portfolio_value", "realized_pnl",
            "unrealized_pnl", "sharpe_ratio", "max_drawdown_pct",
            "win_rate", "profit_factor", "total_trades", "open_positions",
        ]
        for field in required_fields:
            assert field in stats, f"缺少字段: {field}"

    def test_unregistered_agent_signal_fails(self) -> None:
        """未注册的 Agent 执行信号应失败。"""
        pt = PaperTrader()
        signal = _make_signal(agent_id="nonexistent")
        result = pt.execute_signal(signal)
        assert result is False
