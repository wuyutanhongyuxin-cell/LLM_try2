"""Lighter DEX 执行器测试（Mock SDK，不需要真实连接）。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.lighter_executor import LighterExecutor
from src.execution.lighter_helpers import TERMINAL_STATUSES
from src.execution.signal import Action, TradeSignal


def _make_signal(
    action: Action = Action.BUY, size_pct: float = 10.0,
    confidence: float = 0.8,
) -> TradeSignal:
    """快速构建测试用 TradeSignal。"""
    return TradeSignal(
        agent_id="test_agent", agent_name="乐观冲浪型",
        timestamp=datetime.now(tz=timezone.utc),
        action=action, asset="BTC-PERP", size_pct=size_pct,
        entry_price=67000.0, stop_loss_price=65000.0,
        take_profit_price=70000.0, confidence=confidence,
        reasoning="test", personality_influence="test",
        ocean_profile={"openness": 80, "conscientiousness": 20,
                       "extraversion": 80, "agreeableness": 80, "neuroticism": 20},
    )


class TestLighterExecutorInit:
    """初始化与配置测试。"""

    def test_default_values(self) -> None:
        """默认参数正确设置。"""
        ex = LighterExecutor()
        assert ex._max_position == Decimal("0.01")
        assert ex._fill_timeout == 3.0
        assert ex._min_balance == Decimal("10")
        assert ex._dry_run is False
        assert ex._local_position == Decimal("0")
        assert ex._trade_count == 0

    def test_custom_values(self) -> None:
        """自定义参数正确传入。"""
        ex = LighterExecutor(
            max_position=Decimal("0.05"), fill_timeout=5.0,
            min_balance=Decimal("50"), dry_run=True,
        )
        assert ex._max_position == Decimal("0.05")
        assert ex._dry_run is True


class TestDryRun:
    """dry-run 模式测试（不需要真实 SDK）。"""

    @pytest.mark.asyncio
    async def test_dry_run_buy_returns_true(self) -> None:
        """dry-run 模式下 BUY 信号返回 True 但不下单。"""
        ex = LighterExecutor(dry_run=True)
        ex._signer = MagicMock()
        ex._market_index = 0
        signal = _make_signal(Action.BUY)
        # Mock get_balance 返回足够余额
        ex.get_balance = AsyncMock(return_value=Decimal("1000"))
        result = await ex.execute_signal(signal, Decimal("67000"))
        assert result is True
        assert ex._trade_count == 0  # dry-run 不增加计数

    @pytest.mark.asyncio
    async def test_dry_run_hold_returns_false(self) -> None:
        """HOLD 信号始终返回 False。"""
        ex = LighterExecutor(dry_run=True)
        signal = _make_signal(Action.HOLD)
        result = await ex.execute_signal(signal, Decimal("67000"))
        assert result is False

    @pytest.mark.asyncio
    async def test_dry_run_close_all(self) -> None:
        """dry-run 平仓返回 True。"""
        ex = LighterExecutor(dry_run=True)
        ex.get_position = AsyncMock(return_value=Decimal("0.005"))
        result = await ex.close_all_positions()
        assert result is True


class TestExecuteSignal:
    """信号执行逻辑测试。"""

    @pytest.mark.asyncio
    async def test_insufficient_balance(self) -> None:
        """余额不足时拒绝执行。"""
        ex = LighterExecutor(min_balance=Decimal("100"))
        ex.get_balance = AsyncMock(return_value=Decimal("50"))
        signal = _make_signal(Action.BUY)
        result = await ex.execute_signal(signal, Decimal("67000"))
        assert result is False

    @pytest.mark.asyncio
    async def test_max_position_limit(self) -> None:
        """已达最大仓位时拒绝买入。"""
        ex = LighterExecutor(max_position=Decimal("0.01"), dry_run=True)
        ex._local_position = Decimal("0.01")
        ex.get_balance = AsyncMock(return_value=Decimal("1000"))
        signal = _make_signal(Action.BUY)
        result = await ex.execute_signal(signal, Decimal("67000"))
        assert result is False

    @pytest.mark.asyncio
    async def test_sell_no_position(self) -> None:
        """无仓位时卖出返回 False。"""
        ex = LighterExecutor(dry_run=True)
        ex._local_position = Decimal("0")
        ex.get_balance = AsyncMock(return_value=Decimal("1000"))
        signal = _make_signal(Action.SELL)
        result = await ex.execute_signal(signal, Decimal("67000"))
        assert result is False


class TestPositionUpdate:
    """仓位更新测试。"""

    def test_buy_increases_position(self) -> None:
        """买入增加仓位。"""
        ex = LighterExecutor()
        ex._local_position = Decimal("0.005")
        ex._update_position("buy", {"filled_size": Decimal("0.003")})
        assert ex._local_position == Decimal("0.008")

    def test_sell_decreases_position(self) -> None:
        """卖出减少仓位。"""
        ex = LighterExecutor()
        ex._local_position = Decimal("0.005")
        ex._update_position("sell", {"filled_size": Decimal("0.003")})
        assert ex._local_position == Decimal("0.002")

    def test_sell_floor_at_zero(self) -> None:
        """卖出不会低于零。"""
        ex = LighterExecutor()
        ex._local_position = Decimal("0.001")
        ex._update_position("sell", {"filled_size": Decimal("0.005")})
        assert ex._local_position == Decimal("0")


class TestGetAgentStats:
    """绩效统计测试。"""

    def test_stats_format(self) -> None:
        """返回正确的字段。"""
        ex = LighterExecutor()
        ex._local_position = Decimal("0.005")
        ex._trade_count = 3
        stats = ex.get_agent_stats("test")
        assert stats["agent_id"] == "test"
        assert stats["local_position"] == 0.005
        assert stats["trade_count"] == 3


class TestTerminalStatuses:
    """终态集合测试。"""

    def test_filled_is_terminal(self) -> None:
        assert "FILLED" in TERMINAL_STATUSES

    def test_canceled_is_terminal(self) -> None:
        assert "CANCELED" in TERMINAL_STATUSES

    def test_new_is_not_terminal(self) -> None:
        assert "NEW" not in TERMINAL_STATUSES
