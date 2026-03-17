from __future__ import annotations

"""纸上交易执行器：模拟交易执行与绩效跟踪。

管理所有 Agent 的虚拟账户，统一处理信号执行、行情更新、
止损止盈检查，并提供绩效统计和排行榜。
"""

from decimal import Decimal

from loguru import logger

from src.execution.account import AgentAccount
from src.execution.cost_model import CMECostConfig, CostConfig
from src.execution.signal import Action, TradeSignal
from src.execution.stats_helper import (
    calc_max_drawdown_pct,
    calc_profit_factor,
    calc_sharpe_ratio,
    calc_win_rate,
)


class PaperTrader:
    """纸上交易管理器，管理所有 Agent 账户。"""

    def __init__(self, cost_config: CostConfig | None = None) -> None:
        self._accounts: dict[str, AgentAccount] = {}
        self._current_prices: dict[str, float] = {}
        self._cost_config: CostConfig = cost_config or CostConfig()

    def register_agent(
        self, agent_id: str, initial_capital: float,
        cme_cost_config: CMECostConfig | None = None,
        contract_multiplier: float = 1.0,
    ) -> None:
        """注册一个 Agent 虚拟账户。

        Args:
            agent_id: Agent 唯一标识
            initial_capital: 初始资金（float，内部转 Decimal）
            cme_cost_config: CME 成本配置（传入则走 CME 成本路径）
            contract_multiplier: CME 合约乘数（如 ES=50）
        """
        cap = Decimal(str(initial_capital))
        self._accounts[agent_id] = AgentAccount(
            agent_id, cap, self._cost_config,
            cme_cost_config=cme_cost_config,
            contract_multiplier=contract_multiplier,
        )
        logger.info(f"注册 Agent 账户: {agent_id} | 初始资金={cap}")

    def execute_signal(self, signal: TradeSignal) -> bool:
        """执行交易信号。

        Args:
            signal: 交易信号（BUY/SELL/HOLD）

        Returns:
            是否成功执行（HOLD 返回 False）
        """
        account = self._accounts.get(signal.agent_id)
        if account is None:
            logger.error(f"未注册的 Agent: {signal.agent_id}")
            return False
        if signal.action == Action.BUY:
            return account.execute_buy(signal, self._current_prices)
        if signal.action == Action.SELL:
            return account.execute_sell(signal, self._current_prices)
        # HOLD 不执行
        return False

    def update_prices(self, current_prices: dict[str, float]) -> list[dict]:
        """更新行情并检查所有账户的止损/止盈。

        Args:
            current_prices: 各资产最新价格，如 {"BTC-PERP": 67200.0}

        Returns:
            所有触发的平仓事件列表
        """
        self._current_prices = current_prices
        events: list[dict] = []
        for account in self._accounts.values():
            for asset, price in current_prices.items():
                events.extend(account.check_stop_loss_take_profit(asset, price))
        return events

    def get_agent_stats(self, agent_id: str) -> dict:
        """获取单个 Agent 的绩效统计。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            包含 portfolio_value, sharpe_ratio, max_drawdown_pct 等字段的字典
        """
        acc = self._accounts[agent_id]
        pv = acc.get_portfolio_value(self._current_prices)
        return {
            "agent_id": agent_id,
            "portfolio_value": float(pv),
            "realized_pnl": float(acc.get_realized_pnl()),
            "unrealized_pnl": float(acc.get_unrealized_pnl(self._current_prices)),
            "sharpe_ratio": calc_sharpe_ratio(acc.daily_returns),
            "max_drawdown_pct": calc_max_drawdown_pct(acc.peak_value, acc.max_dd_ratio),
            "win_rate": calc_win_rate(acc.closed_trades),
            "profit_factor": calc_profit_factor(acc.closed_trades),
            "total_trades": len(acc.closed_trades),
            "open_positions": len(acc.positions),
            "total_costs": float(acc.total_costs),
        }

    def get_leaderboard(self) -> list[dict]:
        """按 Sharpe Ratio 降序返回所有 Agent 的绩效排行。"""
        stats = [self.get_agent_stats(aid) for aid in self._accounts]
        stats.sort(key=lambda s: s["sharpe_ratio"], reverse=True)
        return stats

    def record_daily_returns(self) -> None:
        """为所有账户记录一次日收益率（每天调用一次）。"""
        for account in self._accounts.values():
            account.record_daily_return(self._current_prices)
