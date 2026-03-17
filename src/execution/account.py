from __future__ import annotations

"""Agent 虚拟账户：持仓模型与账户逻辑。

每个 Agent 拥有独立的 AgentAccount，互不共享。
所有金额计算使用 Decimal，禁止 float 算钱。
"""

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
        cme_cost_config: CMECostConfig | None = None,
        contract_multiplier: float = 1.0,
    ) -> None:
        self.agent_id: str = agent_id
        self.initial_capital: Decimal = initial_capital
        self.cash: Decimal = initial_capital
        self.positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.daily_returns: list[float] = []
        self.peak_value: Decimal = initial_capital
        self.max_dd_ratio: float = 0.0
        self._last_portfolio_value: Decimal = initial_capital
        self._cost_config: CostConfig = cost_config or CostConfig()
        self._cme_cost_config: CMECostConfig | None = cme_cost_config
        self._contract_multiplier: float = contract_multiplier
        self._is_cme: bool = cme_cost_config is not None
        self.total_costs: Decimal = Decimal("0")  # 累计交易成本

    def execute_buy(self, signal: TradeSignal, current_prices: dict[str, float]) -> bool:
        """执行买入信号，开多仓。

        Args:
            signal: 交易信号
            current_prices: 当前各资产价格

        Returns:
            是否成功执行
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        notional = portfolio_value * Decimal(str(signal.size_pct)) / Decimal("100")
        # A1: 计算开仓成本（区分 crypto / CME 路径）
        if self._is_cme and self._cme_cost_config is not None:
            contracts = max(1, int(float(notional) / (
                signal.entry_price * self._contract_multiplier)))
            cost_result = calculate_cme_entry_cost(
                signal.entry_price, contracts, self._contract_multiplier,
                "LONG", self._cme_cost_config)
        else:
            cost_result = calculate_entry_cost(
                signal.entry_price, float(notional), "LONG", self._cost_config)
        total_needed = notional + Decimal(str(cost_result.total_cost))
        if total_needed > self.cash:
            logger.warning(f"[{self.agent_id}] 资金不足: 需要 {total_needed}, 可用 {self.cash}")
            return False
        self.cash -= total_needed
        self.total_costs += Decimal(str(cost_result.total_cost))
        pos = Position(
            agent_id=self.agent_id,
            asset=signal.asset,
            side="LONG",
            size_pct=signal.size_pct,
            entry_price=Decimal(str(cost_result.effective_price)),
            stop_loss_price=_to_decimal(signal.stop_loss_price),
            take_profit_price=_to_decimal(signal.take_profit_price),
            opened_at=signal.timestamp,
            notional=notional,
        )
        self.positions.append(pos)
        logger.info(f"[{self.agent_id}] 开多 {signal.asset} | 金额={notional} | "
                     f"入场={cost_result.effective_price} | 成本={cost_result.total_cost:.4f}")
        return True

    def execute_sell(self, signal: TradeSignal, current_prices: dict[str, float]) -> bool:
        """执行卖出信号，平掉对应资产的多仓。"""
        pos = self._find_position(signal.asset)
        if pos is None:
            logger.warning(f"[{self.agent_id}] 无 {signal.asset} 持仓可平")
            return False
        self._close_position(pos, Decimal(str(signal.entry_price)), "SIGNAL_SELL")
        return True

    def check_stop_loss_take_profit(
        self, asset: str, current_price: float
    ) -> list[dict]:
        """检查指定资产持仓是否触发止损或止盈（等于也触发）。"""
        price = Decimal(str(current_price))
        events: list[dict] = []
        for pos in list(self.positions):  # 遍历副本，_close_position 会修改列表
            if pos.asset != asset:
                continue
            if pos.side == "LONG":
                if pos.stop_loss_price is not None and price <= pos.stop_loss_price:
                    events.append(self._close_position(pos, price, "STOP_LOSS"))
                elif pos.take_profit_price is not None and price >= pos.take_profit_price:
                    events.append(self._close_position(pos, price, "TAKE_PROFIT"))
        return events

    def get_portfolio_value(self, current_prices: dict[str, float]) -> Decimal:
        """计算总资产 = 可用资金 + 持仓市值。"""
        return self.cash + self._positions_value(current_prices)

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> Decimal:
        """计算所有持仓的未实现盈亏。"""
        pnl = Decimal("0")
        for pos in self.positions:
            cur = Decimal(str(current_prices.get(pos.asset, 0)))
            if pos.entry_price == Decimal("0"):
                continue
            pnl += pos.notional * (cur - pos.entry_price) / pos.entry_price
        return pnl

    def get_realized_pnl(self) -> Decimal:
        """累计已实现盈亏。"""
        return sum((t["pnl"] for t in self.closed_trades), Decimal("0"))

    def record_daily_return(self, current_prices: dict[str, float]) -> None:
        """记录一次日收益率，并更新峰值和最大回撤。"""
        value = self.get_portfolio_value(current_prices)
        if self._last_portfolio_value > Decimal("0"):
            ret = float((value - self._last_portfolio_value) / self._last_portfolio_value)
            self.daily_returns.append(ret)
        self._last_portfolio_value = value
        if value > self.peak_value:
            self.peak_value = value
        if self.peak_value > Decimal("0"):
            dd_ratio = float((value - self.peak_value) / self.peak_value)
            if dd_ratio < self.max_dd_ratio:
                self.max_dd_ratio = dd_ratio

    def _find_position(self, asset: str) -> Position | None:
        """查找指定资产的持仓（返回第一个匹配）。"""
        for p in self.positions:
            if p.asset == asset:
                return p
        return None

    def _close_position(self, pos: Position, close_price: Decimal, reason: str) -> dict:
        """平仓并记录交易（含平仓成本）。"""
        # A1: 计算平仓成本（区分 crypto / CME 路径）
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
        effective_close = Decimal(str(exit_cost.effective_price))
        if pos.entry_price > Decimal("0"):
            pnl = pos.notional * (effective_close - pos.entry_price) / pos.entry_price - cost_dec
        else:
            pnl = Decimal("0")
        self.cash += pos.notional + pnl
        self.total_costs += cost_dec
        self.positions.remove(pos)
        trade_record = {
            "agent_id": self.agent_id,
            "asset": pos.asset,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "close_price": close_price,
            "notional": pos.notional,
            "pnl": pnl,
            "reason": reason,
            "opened_at": pos.opened_at,
            "closed_at": datetime.now(tz=timezone.utc),
        }
        self.closed_trades.append(trade_record)
        logger.info(f"[{self.agent_id}] 平仓 {pos.asset} | 原因={reason} | PnL={pnl:.4f}")
        return trade_record

    def _positions_value(self, current_prices: dict[str, float]) -> Decimal:
        """计算全部持仓当前市值。"""
        total = Decimal("0")
        for pos in self.positions:
            cur = Decimal(str(current_prices.get(pos.asset, 0)))
            if pos.entry_price > Decimal("0"):
                total += pos.notional * cur / pos.entry_price
            else:
                total += pos.notional
        return total


def _to_decimal(value: float | None) -> Decimal | None:
    """将 float 转为 Decimal，None 保持 None。"""
    return Decimal(str(value)) if value is not None else None
