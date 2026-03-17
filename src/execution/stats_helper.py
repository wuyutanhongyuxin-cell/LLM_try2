from __future__ import annotations

"""绩效统计辅助函数：Sharpe Ratio、最大回撤、胜率、盈亏比。

从 paper_trader.py 拆分出来，避免单文件超 200 行。
"""

import math
from decimal import Decimal


def calc_sharpe_ratio(daily_returns: list[float]) -> float:
    """计算年化 Sharpe Ratio（基于日收益率序列）。

    公式: mean(returns) / std(returns) * sqrt(365)
    如果标准差为 0 或数据不足，返回 0.0。

    Args:
        daily_returns: 日收益率列表（小数形式，如 0.01 = 1%）

    Returns:
        年化 Sharpe Ratio
    """
    if len(daily_returns) < 2:
        return 0.0
    n = len(daily_returns)
    mean_r = sum(daily_returns) / n
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
    std_r = math.sqrt(variance)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(365)


def calc_max_drawdown_pct(peak_value: Decimal, current_min_ratio: float) -> float:
    """返回已记录的最大回撤百分比（负值）。

    Args:
        peak_value: 历史最高净值（保留但未使用，回撤在 account 中实时跟踪）
        current_min_ratio: 已跟踪的最小 (value - peak) / peak 比值

    Returns:
        最大回撤百分比（如 -3.2 表示 3.2%）
    """
    return round(current_min_ratio * 100, 2)


def calc_win_rate(closed_trades: list[dict]) -> float:
    """计算胜率：盈利交易数 / 总交易数。

    Args:
        closed_trades: 已平仓交易记录列表，每条包含 'pnl' 字段

    Returns:
        胜率 0.0-1.0，无交易时返回 0.0
    """
    if not closed_trades:
        return 0.0
    wins = sum(1 for t in closed_trades if t["pnl"] > Decimal("0"))
    return round(wins / len(closed_trades), 4)


def calc_profit_factor(closed_trades: list[dict]) -> float:
    """计算盈亏比：总盈利 / 总亏损的绝对值。

    Args:
        closed_trades: 已平仓交易记录列表，每条包含 'pnl' 字段

    Returns:
        盈亏比，无亏损时返回 0.0
    """
    total_profit = sum(t["pnl"] for t in closed_trades if t["pnl"] > Decimal("0"))
    total_loss = sum(abs(t["pnl"]) for t in closed_trades if t["pnl"] < Decimal("0"))
    if total_loss == Decimal("0"):
        return 0.0
    return round(float(total_profit / total_loss), 4)
