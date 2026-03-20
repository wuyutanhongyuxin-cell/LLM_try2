from __future__ import annotations
"""Agent 虚拟账户：持仓模型（LONG/SHORT 双向）与账户逻辑。"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.execution.cost_model import (
    CMECostConfig, CostConfig,
    calculate_cme_entry_cost, calculate_cme_exit_cost,
    calculate_entry_cost, calculate_exit_cost,
)
from src.execution.signal import TradeSignal


class Position(BaseModel):
    """持仓记录。"""

    agent_id: str = Field(..., description="Agent 标识")
    asset: str = Field(..., description="资产标识")
    side: str = Field(..., description="方向: LONG/SHORT")
    size_pct: float = Field(..., description="仓位占比")
    entry_price: Decimal = Field(..., description="入场价格")
    stop_loss_price: Optional[Decimal] = Field(None, description="止损价格")
    take_profit_price: Optional[Decimal] = Field(None, description="止盈价格")
    opened_at: datetime = Field(..., description="开仓时间")
    notional: Decimal = Field(..., description="名义金额（入场时锁定）")


class AgentAccount:
    """单个 Agent 的虚拟交易账户。"""

    def __init__(
        self, agent_id: str, initial_capital: Decimal,
        cost_config: CostConfig | None = None,
        cme_cost_config: CMECostConfig | None = None, contract_multiplier: float = 1.0,
    ) -> None:
        self.agent_id, self.initial_capital = agent_id, initial_capital
        self.cash: Decimal = initial_capital
        self.positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.daily_returns: list[float] = []
        self.peak_value = self._last_portfolio_value = initial_capital
        self.max_dd_ratio: float = 0.0
        self.total_costs: Decimal = Decimal("0")
        self._cost_config = cost_config or CostConfig()
        self._cme_cost_config = cme_cost_config
        self._contract_multiplier = contract_multiplier
        self._is_cme = cme_cost_config is not None

    def execute_buy(self, signal: TradeSignal, current_prices: dict[str, float]) -> bool:
        """执行买入：持有空仓→平空，否则→开多。"""
        # 如果持有空仓，先平空（买入平仓）
        existing = self._find_position(signal.asset)
        if existing and existing.side == "SHORT":
            self._close_position(existing, Decimal(str(signal.entry_price)), "SIGNAL_BUY")
            return True
        return self._open_position(signal, current_prices, "LONG")

    def execute_sell(self, signal: TradeSignal, current_prices: dict[str, float]) -> bool:
        """执行卖出：持有多仓→平多，否则→开空。"""
        existing = self._find_position(signal.asset)
        if existing and existing.side == "LONG":
            self._close_position(existing, Decimal(str(signal.entry_price)), "SIGNAL_SELL")
            return True
        return self._open_position(signal, current_prices, "SHORT")

    def _open_position(
        self, signal: TradeSignal, current_prices: dict[str, float], side: str,
    ) -> bool:
        """开仓通用逻辑（LONG/SHORT 共用）。"""
        portfolio_value = self.get_portfolio_value(current_prices)
        notional = portfolio_value * Decimal(str(signal.size_pct)) / Decimal("100")
        if self._is_cme and self._cme_cost_config is not None:
            contracts = max(1, int(float(notional) / (
                signal.entry_price * self._contract_multiplier)))
            cost_result = calculate_cme_entry_cost(
                signal.entry_price, contracts, self._contract_multiplier,
                side, self._cme_cost_config)
        else:
            cost_result = calculate_entry_cost(
                signal.entry_price, float(notional), side, self._cost_config)
        total_needed = notional + Decimal(str(cost_result.total_cost))
        if total_needed > self.cash:
            logger.warning(f"[{self.agent_id}] 资金不足: 需要 {total_needed}, 可用 {self.cash}")
            return False
        self.cash -= total_needed
        self.total_costs += Decimal(str(cost_result.total_cost))
        pos = Position(
            agent_id=self.agent_id, asset=signal.asset, side=side,
            size_pct=signal.size_pct,
            entry_price=Decimal(str(cost_result.effective_price)),
            stop_loss_price=_to_decimal(signal.stop_loss_price),
            take_profit_price=_to_decimal(signal.take_profit_price),
            opened_at=signal.timestamp, notional=notional,
        )
        self.positions.append(pos)
        label = "开多" if side == "LONG" else "开空"
        logger.info(f"[{self.agent_id}] {label} {signal.asset} | 金额={notional} | "
                     f"入场={cost_result.effective_price} | 成本={cost_result.total_cost:.4f}")
        return True

    def check_stop_loss_take_profit(self, asset: str, current_price: float) -> list[dict]:
        """检查 SL/TP：LONG SL=价跌, SHORT SL=价涨（等于也触发）。"""
        price, events = Decimal(str(current_price)), []
        for pos in list(self.positions):
            if pos.asset != asset:
                continue
            sl, tp = pos.stop_loss_price, pos.take_profit_price
            # LONG: SL 在下方，TP 在上方；SHORT: SL 在上方，TP 在下方
            sl_hit = sl is not None and (price <= sl if pos.side == "LONG" else price >= sl)
            tp_hit = tp is not None and (price >= tp if pos.side == "LONG" else price <= tp)
            if sl_hit:
                events.append(self._close_position(pos, price, "STOP_LOSS"))
            elif tp_hit:
                events.append(self._close_position(pos, price, "TAKE_PROFIT"))
        return events

    def get_portfolio_value(self, current_prices: dict[str, float]) -> Decimal:
        return self.cash + self._positions_value(current_prices)

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> Decimal:
        pnl = Decimal("0")
        for pos in self.positions:
            cur = Decimal(str(current_prices.get(pos.asset, 0)))
            if pos.entry_price == Decimal("0"):
                continue
            diff = (cur - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - cur)
            pnl += pos.notional * diff / pos.entry_price
        return pnl

    def get_realized_pnl(self) -> Decimal:
        return sum((t["pnl"] for t in self.closed_trades), Decimal("0"))

    def record_daily_return(self, current_prices: dict[str, float]) -> None:
        """记录日收益率，更新峰值和最大回撤。"""
        value = self.get_portfolio_value(current_prices)
        if self._last_portfolio_value > Decimal("0"):
            self.daily_returns.append(
                float((value - self._last_portfolio_value) / self._last_portfolio_value))
        self._last_portfolio_value = value
        if value > self.peak_value:
            self.peak_value = value
        if self.peak_value > Decimal("0"):
            dd = float((value - self.peak_value) / self.peak_value)
            self.max_dd_ratio = min(self.max_dd_ratio, dd)

    def _find_position(self, asset: str) -> Position | None:
        return next((p for p in self.positions if p.asset == asset), None)

    def _close_position(self, pos: Position, close_price: Decimal, reason: str) -> dict:
        """平仓并记录交易（含平仓成本，区分 LONG/SHORT PnL）。"""
        if self._is_cme and self._cme_cost_config is not None:
            contracts = max(1, int(float(pos.notional) / (
                float(close_price) * self._contract_multiplier)))
            exit_cost = calculate_cme_exit_cost(
                float(close_price), contracts, self._contract_multiplier,
                pos.side, self._cme_cost_config)
        else:
            exit_cost = calculate_exit_cost(
                float(close_price), float(pos.notional), pos.side, self._cost_config)
        cost_dec = Decimal(str(exit_cost.total_cost))
        eff = Decimal(str(exit_cost.effective_price))
        if pos.entry_price > Decimal("0"):
            # LONG: (close-entry)/entry; SHORT: (entry-close)/entry
            diff = (eff - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - eff)
            pnl = pos.notional * diff / pos.entry_price - cost_dec
        else:
            pnl = Decimal("0")
        self.cash += pos.notional + pnl
        self.total_costs += cost_dec
        self.positions.remove(pos)
        record = {
            "agent_id": self.agent_id, "asset": pos.asset, "side": pos.side,
            "entry_price": pos.entry_price, "close_price": close_price,
            "notional": pos.notional, "pnl": pnl, "reason": reason,
            "opened_at": pos.opened_at, "closed_at": datetime.now(tz=timezone.utc),
        }
        self.closed_trades.append(record)
        logger.info(f"[{self.agent_id}] 平仓 {pos.asset} | {pos.side} | {reason} | PnL={pnl:.4f}")
        return record

    def _positions_value(self, current_prices: dict[str, float]) -> Decimal:
        total = Decimal("0")
        for pos in self.positions:
            cur = Decimal(str(current_prices.get(pos.asset, 0)))
            if pos.entry_price <= Decimal("0"):
                total += pos.notional
            elif pos.side == "LONG":
                total += pos.notional * cur / pos.entry_price
            else:  # SHORT: notional × (2×entry - current) / entry
                total += pos.notional * (2 * pos.entry_price - cur) / pos.entry_price
        return total


def _to_decimal(value: float | None) -> Decimal | None:
    """将 float 转为 Decimal，None 保持 None。"""
    return Decimal(str(value)) if value is not None else None
