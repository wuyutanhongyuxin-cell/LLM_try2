"""全局风控管理器。

跨所有 Agent 监控全局回撤和单日亏损，超阈值时暂停全部交易。
"""

from __future__ import annotations

from decimal import Decimal

from loguru import logger


class RiskManager:
    """全局风控，检查跨 Agent 的风险指标。"""

    def __init__(self, global_max_drawdown_pct: float = 25.0,
                 global_max_daily_loss_pct: float = 10.0) -> None:
        self._max_drawdown = Decimal(str(global_max_drawdown_pct))
        self._max_daily_loss = Decimal(str(global_max_daily_loss_pct))
        self._initial_total_value: Decimal | None = None
        self._peak_total_value: Decimal | None = None
        self._daily_start_value: Decimal | None = None
        self._is_halted: bool = False
        self._halt_reason: str | None = None

    def initialize(self, total_value: Decimal) -> None:
        """初始化全局风控基准值（系统启动时调用一次）。"""
        self._initial_total_value = total_value
        self._peak_total_value = total_value
        self._daily_start_value = total_value
        self._is_halted = False
        self._halt_reason = None
        logger.info("风控初始化 | 基准总资产={}", total_value)

    def reset_daily(self, current_total_value: Decimal) -> None:
        """每日重置（新交易日开始时调用）。仅解除单日亏损暂停。"""
        self._daily_start_value = current_total_value
        if self._is_halted and self._halt_reason == "daily_loss":
            self._is_halted = False
            self._halt_reason = None
            logger.info("风控 | 单日亏损暂停已解除，新交易日开始")

    def check_risk(self, current_total_value: Decimal) -> dict:
        """检查全局风险指标，返回状态字典。"""
        if self._peak_total_value is None:
            return {"is_halted": False, "reason": None, "drawdown_pct": 0.0, "daily_loss_pct": 0.0}
        # 更新峰值
        if current_total_value > self._peak_total_value:
            self._peak_total_value = current_total_value
        dd = self._pct_loss(self._peak_total_value, current_total_value)
        dl = self._pct_loss(self._daily_start_value, current_total_value)
        # 全局回撤检查
        if dd >= self._max_drawdown and not self._is_halted:
            self._is_halted, self._halt_reason = True, "max_drawdown"
            logger.error("风控触发 | 全局回撤={:.2f}% 阈值={}%", dd, self._max_drawdown)
        # 单日亏损检查
        if dl >= self._max_daily_loss and not self._is_halted:
            self._is_halted, self._halt_reason = True, "daily_loss"
            logger.error("风控触发 | 当日亏损={:.2f}% 阈值={}%", dl, self._max_daily_loss)
        return {"is_halted": self._is_halted, "reason": self._halt_reason,
                "drawdown_pct": float(dd), "daily_loss_pct": float(dl)}

    def can_trade(self) -> bool:
        """是否允许继续交易。"""
        return not self._is_halted

    @property
    def is_halted(self) -> bool:
        """当前是否处于暂停状态。"""
        return self._is_halted

    # ── E1: Agent 级风控 ────────────────────────────────

    def check_agent_risk(self, agent_id: str, agent_stats: dict) -> dict:
        """检查单个 Agent 的风险状态。

        触发条件：
        - 已实现亏损 > max_drawdown_pct → 暂停该 Agent
        - 连续亏损 >= 5 笔 → 告警
        - 单笔亏损 > 总资产 10% → 告警
        """
        alerts: list[str] = []
        should_halt = False
        dd = abs(agent_stats.get("max_drawdown_pct", 0))
        if dd > float(self._max_drawdown):
            alerts.append(f"回撤 {dd:.1f}% > 阈值 {self._max_drawdown}%")
            should_halt = True
        pnl = agent_stats.get("realized_pnl", 0)
        pv = agent_stats.get("portfolio_value", 1)
        if pv > 0 and pnl < 0 and abs(pnl) / pv * 100 > float(self._max_drawdown):
            alerts.append(f"累计亏损 {abs(pnl):.0f} 超限")
            should_halt = True
        return {
            "agent_id": agent_id,
            "should_halt": should_halt,
            "alerts": alerts,
        }

    @staticmethod
    def _pct_loss(baseline: Decimal | None, current: Decimal) -> Decimal:
        """计算从 baseline 到 current 的亏损百分比（正值=亏损）。"""
        if not baseline or baseline == Decimal("0"):
            return Decimal("0")
        loss = baseline - current
        return (loss / baseline) * Decimal("100") if loss > 0 else Decimal("0")
