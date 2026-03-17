from __future__ import annotations

"""Agent 行为一致性监控器测试。"""

import pytest

from src.execution.consistency_monitor import ConsistencyMonitor, kl_divergence


def _make_signal(action: str = "BUY", size_pct: float = 10.0, confidence: float = 0.8) -> dict:
    """创建测试信号。"""
    return {"action": action, "size_pct": size_pct, "confidence": confidence}


class TestKLDivergence:
    """KL 散度计算测试。"""

    def test_kl_identical_distributions(self) -> None:
        """相同分布的 KL 散度应为 0。"""
        p = {"BUY": 0.5, "SELL": 0.3, "HOLD": 0.2}
        q = {"BUY": 0.5, "SELL": 0.3, "HOLD": 0.2}
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-8)

    def test_kl_different_distributions(self) -> None:
        """不同分布的 KL 散度应 > 0。"""
        p = {"BUY": 0.8, "SELL": 0.1, "HOLD": 0.1}
        q = {"BUY": 0.3, "SELL": 0.4, "HOLD": 0.3}
        assert kl_divergence(p, q) > 0


class TestConsistencyMonitor:
    """一致性监控器测试。"""

    def test_no_baseline_no_drift(self) -> None:
        """未设基线时 is_drifting 应为 False。"""
        monitor = ConsistencyMonitor(window_size=10)
        result = monitor.check(_make_signal())
        assert result["is_drifting"] is False

    def test_warming_up_no_drift(self) -> None:
        """窗口未填满一半时不应误报漂移。"""
        monitor = ConsistencyMonitor(window_size=20)
        # 设基线：全 BUY
        baseline = [_make_signal("BUY") for _ in range(30)]
        monitor.set_baseline(baseline)
        # 只发 5 条完全不同的信号（< 窗口一半 10）
        for _ in range(5):
            result = monitor.check(_make_signal("SELL"))
        assert result["is_drifting"] is False

    def test_no_drift_normal_signals(self) -> None:
        """与基线一致的信号不应检测到漂移。"""
        monitor = ConsistencyMonitor(window_size=20)
        # 基线：均匀分布
        baseline = (
            [_make_signal("BUY") for _ in range(10)]
            + [_make_signal("SELL") for _ in range(10)]
            + [_make_signal("HOLD") for _ in range(10)]
        )
        monitor.set_baseline(baseline)
        # 窗口内保持同样分布
        for _ in range(7):
            monitor.check(_make_signal("BUY"))
        for _ in range(7):
            monitor.check(_make_signal("SELL"))
        for i in range(7):
            result = monitor.check(_make_signal("HOLD"))
        assert result["severity"] == "normal"

    def test_action_drift_detected(self) -> None:
        """action 分布突变时应检测到漂移。"""
        monitor = ConsistencyMonitor(window_size=20)
        # 基线：全 BUY
        baseline = [_make_signal("BUY") for _ in range(30)]
        monitor.set_baseline(baseline)
        # 窗口全填 SELL（与基线完全相反）
        for i in range(20):
            result = monitor.check(_make_signal("SELL"))
        assert result["is_drifting"] is True
        assert result["action_kl"] > 0

    def test_three_tier_severity(self) -> None:
        """验证 warning / critical / halt 三级阈值。"""
        # 使用小窗口加速填满
        for threshold, expected_severity in [
            (0.01, "warning"),    # 很低的阈值 → 容易触发 warning
            (0.01, "critical"),   # 更大偏移 → critical
            (0.01, "halt"),       # 极大偏移 → halt
        ]:
            pass  # 用分开的子测试更清晰

        # -- warning: 轻微偏移 --
        monitor_w = ConsistencyMonitor(
            window_size=10, warning_threshold=0.05, critical_threshold=5.0, halt_threshold=10.0
        )
        baseline_w = [_make_signal("BUY") for _ in range(5)] + [_make_signal("SELL") for _ in range(5)]
        monitor_w.set_baseline(baseline_w)
        # 窗口偏向 BUY（7:3 vs 5:5）
        for _ in range(7):
            monitor_w.check(_make_signal("BUY"))
        for _ in range(3):
            result_w = monitor_w.check(_make_signal("SELL"))
        assert result_w["severity"] == "warning"

        # -- critical: 中等偏移 --
        monitor_c = ConsistencyMonitor(
            window_size=10, warning_threshold=0.01, critical_threshold=0.1, halt_threshold=100.0
        )
        baseline_c = [_make_signal("HOLD") for _ in range(10)]
        monitor_c.set_baseline(baseline_c)
        # 窗口全 BUY（极大偏移，但 halt 阈值设得极高不会触发）
        for _ in range(10):
            result_c = monitor_c.check(_make_signal("BUY"))
        assert result_c["severity"] == "critical"

        # -- halt: 极端偏移 --
        monitor_h = ConsistencyMonitor(
            window_size=10, warning_threshold=0.01, critical_threshold=0.05, halt_threshold=0.5
        )
        baseline_h = [_make_signal("HOLD") for _ in range(10)]
        monitor_h.set_baseline(baseline_h)
        for _ in range(10):
            result_h = monitor_h.check(_make_signal("BUY"))
        assert result_h["severity"] == "halt"
