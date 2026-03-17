from __future__ import annotations
"""Agent 行为一致性监控器。

对比当前行为窗口 vs 基线分布，检测漂移信号：
1. Action 分布漂移（BUY/SELL/HOLD 比例突变）— KL 散度
2. 仓位大小漂移（平均 size_pct 偏离基线）
3. 置信度分布漂移（平均 confidence 偏离基线）

三级阈值体系（基于 Evidently AI + PSI 标准）：
- warning: KL > 0.1（记录日志）
- critical: KL > 0.2（Telegram 告警）
- halt: KL > 0.5（暂停该 Agent）
"""

import math

from loguru import logger


def kl_divergence(p: dict[str, float], q: dict[str, float], epsilon: float = 1e-10) -> float:
    """计算 KL(P || Q)，对 0 概率加 epsilon 平滑。"""
    result = 0.0
    all_keys = set(p.keys()) | set(q.keys())
    for key in all_keys:
        p_val = max(p.get(key, 0.0), epsilon)
        q_val = max(q.get(key, 0.0), epsilon)
        result += p_val * math.log(p_val / q_val)
    return result


class ConsistencyMonitor:
    """Agent 行为一致性监控器。"""

    def __init__(
        self,
        window_size: int = 50,
        warning_threshold: float = 0.1,
        critical_threshold: float = 0.2,
        halt_threshold: float = 0.5,
    ) -> None:
        self._window_size = window_size
        self._warning = warning_threshold
        self._critical = critical_threshold
        self._halt = halt_threshold
        self._baseline_action_dist: dict[str, float] = {}
        self._baseline_avg_size: float = 0.0
        self._baseline_avg_conf: float = 0.0
        self._recent_signals: list[dict] = []
        self._has_baseline: bool = False

    def set_baseline(self, signals: list[dict]) -> None:
        """从历史信号建立基线分布。"""
        if not signals:
            return
        self._baseline_action_dist = _action_distribution(signals)
        self._baseline_avg_size = _avg_field(signals, "size_pct")
        self._baseline_avg_conf = _avg_field(signals, "confidence")
        self._has_baseline = True

    def check(self, signal: dict) -> dict:
        """检查新信号是否导致行为漂移。"""
        self._recent_signals.append(signal)
        if len(self._recent_signals) > self._window_size:
            self._recent_signals = self._recent_signals[-self._window_size:]
        # 基线未建立或窗口未填满一半时不检测
        if not self._has_baseline or len(self._recent_signals) < self._window_size // 2:
            return {"is_drifting": False, "action_kl": 0.0, "size_drift_pct": 0.0,
                    "confidence_drift_pct": 0.0, "severity": "normal", "alert_reasons": []}
        current_dist = _action_distribution(self._recent_signals)
        action_kl = kl_divergence(current_dist, self._baseline_action_dist)
        size_drift = _pct_change(_avg_field(self._recent_signals, "size_pct"), self._baseline_avg_size)
        conf_drift = _pct_change(_avg_field(self._recent_signals, "confidence"), self._baseline_avg_conf)
        # 判断严重程度
        alerts: list[str] = []
        severity = "normal"
        if action_kl > self._halt:
            severity = "halt"
            alerts.append(f"Action KL={action_kl:.3f} > halt({self._halt})")
        elif action_kl > self._critical:
            severity = "critical"
            alerts.append(f"Action KL={action_kl:.3f} > critical({self._critical})")
        elif action_kl > self._warning:
            severity = "warning"
            alerts.append(f"Action KL={action_kl:.3f} > warning({self._warning})")
        if abs(size_drift) > 50:
            alerts.append(f"Size drift {size_drift:.1f}%")
        if abs(conf_drift) > 30:
            alerts.append(f"Confidence drift {conf_drift:.1f}%")
        is_drifting = severity != "normal" or len(alerts) > 0
        return {
            "is_drifting": is_drifting,
            "action_kl": round(action_kl, 4),
            "size_drift_pct": round(size_drift, 2),
            "confidence_drift_pct": round(conf_drift, 2),
            "severity": severity,
            "alert_reasons": alerts,
        }


def _action_distribution(signals: list[dict]) -> dict[str, float]:
    """计算 action 分布（归一化）。"""
    counts: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for s in signals:
        action = str(s.get("action", "HOLD")).upper()
        counts[action] = counts.get(action, 0) + 1
    total = max(sum(counts.values()), 1)
    return {k: v / total for k, v in counts.items()}


def _avg_field(signals: list[dict], field: str) -> float:
    """计算信号列表中指定字段的平均值。"""
    vals = [s.get(field, 0) for s in signals if field in s]
    return sum(vals) / max(len(vals), 1)


def _pct_change(current: float, baseline: float) -> float:
    """计算百分比变化。"""
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline * 100
