from __future__ import annotations

"""对抗性场景生成器测试。"""

import pytest

from src.market.adversarial import (
    SCENARIOS,
    generate_adversarial_prices,
    inject_adversarial,
)


BASE_PRICE = 67000.0


class TestGenerateAdversarialPrices:
    """generate_adversarial_prices 测试。"""

    def test_flash_crash_min_below_85pct(self) -> None:
        """闪崩场景最低价 < base_price × 0.85。"""
        prices = generate_adversarial_prices(BASE_PRICE, "flash_crash")
        assert min(prices) < BASE_PRICE * 0.85

    def test_pump_max_above_110pct(self) -> None:
        """暴涨场景最高价 > base_price × 1.10。"""
        prices = generate_adversarial_prices(BASE_PRICE, "pump")
        assert max(prices) > BASE_PRICE * 1.10

    def test_sideways_within_5pct_band(self) -> None:
        """横盘场景所有价格在 base_price × [0.95, 1.05] 之间。"""
        prices = generate_adversarial_prices(BASE_PRICE, "sideways", seed=42)
        for p in prices:
            assert BASE_PRICE * 0.95 <= p <= BASE_PRICE * 1.05

    def test_correct_length(self) -> None:
        """每种场景生成的价格数量与配置一致。"""
        for scenario, cfg in SCENARIOS.items():
            prices = generate_adversarial_prices(BASE_PRICE, scenario, seed=42)
            assert len(prices) == cfg["bars"], f"{scenario}: 期望 {cfg['bars']}，实际 {len(prices)}"

    def test_same_seed_same_output(self) -> None:
        """相同 seed 输出一致（sideways 场景依赖随机数）。"""
        a = generate_adversarial_prices(BASE_PRICE, "sideways", seed=123)
        b = generate_adversarial_prices(BASE_PRICE, "sideways", seed=123)
        assert a == b


class TestInjectAdversarial:
    """inject_adversarial 测试。"""

    def test_length_unchanged(self) -> None:
        """注入后列表总长度不变。"""
        original = [67000.0 + i * 100 for i in range(200)]
        result = inject_adversarial(original, "flash_crash", inject_at=50)
        assert len(result) == len(original)

    def test_before_inject_unchanged(self) -> None:
        """注入位置之前的价格不变。"""
        original = [67000.0 + i * 100 for i in range(200)]
        result = inject_adversarial(original, "flash_crash", inject_at=50)
        assert result[:50] == original[:50]

    def test_after_inject_unchanged(self) -> None:
        """注入范围之后的价格不变。"""
        original = [67000.0 + i * 100 for i in range(200)]
        bars = SCENARIOS["flash_crash"]["bars"]
        result = inject_adversarial(original, "flash_crash", inject_at=50)
        assert result[50 + bars:] == original[50 + bars:]

    def test_inject_at_boundary(self) -> None:
        """注入位置在末尾附近不会越界。"""
        original = [67000.0] * 10
        result = inject_adversarial(original, "flash_crash", inject_at=8)
        assert len(result) == 10
