"""对抗性行情场景生成器。

基于真实 BTC 历史极端事件，注入到 MockDataFeed 中测试 Agent 鲁棒性。
参考：TradeTrap (arxiv.org/abs/2512.02261) 的对抗性压力测试思路。

场景数据来源（经联网验证）:
- flash_crash: 2024.3.19 BitMEX闪崩，BTC从$67K跌至$8.9K（仅BitMEX现货）
- pump: 2024.12.5 BTC突破$100K
- fake_breakout: 2024 Q1 Grayscale GBTC抛售引发的先涨后跌
- sideways: 2023 Q3 BTC在$25K-$30K区间震荡50天
- v_reversal: 2024.12 BTC $100K→$93K→$100K V型反转
"""

from __future__ import annotations

import random
from typing import Literal

# 场景类型
ScenarioType = Literal["flash_crash", "pump", "fake_breakout", "sideways", "v_reversal"]

# 5 种场景的预设参数（基于真实 BTC 历史数据验证）
SCENARIOS: dict[str, dict] = {
    "flash_crash": {
        "description": "闪崩：模拟2024.3 BitMEX事件，单根K线暴跌",
        "bars": 5,
        "drops": [-0.15, -0.05, -0.03, 0.02, 0.08],
    },
    "pump": {
        "description": "暴涨：模拟2024.12突破$100K",
        "bars": 5,
        "drops": [0.05, 0.06, 0.04, 0.02, -0.02],
    },
    "fake_breakout": {
        "description": "假突破：先涨后跌，模拟2024 Q1 Grayscale抛售期",
        "bars": 6,
        "drops": [0.03, 0.02, -0.04, -0.04, -0.02, 0.01],
    },
    "sideways": {
        "description": "极端横盘：模拟2023 Q3区间震荡",
        "bars": 50,
        "drops": None,  # 随机生成 ±1%
    },
    "v_reversal": {
        "description": "V型反转：模拟2024.12.5事件",
        "bars": 6,
        "drops": [-0.03, -0.02, -0.01, 0.02, 0.025, 0.02],
    },
}


def generate_adversarial_prices(
    base_price: float,
    scenario: ScenarioType,
    seed: int | None = None,
) -> list[float]:
    """根据场景生成对抗性价格序列。

    Args:
        base_price: 起始价格
        scenario: 场景类型
        seed: 随机种子（sideways 场景用）

    Returns:
        价格列表（长度 = SCENARIOS[scenario]["bars"]）
    """
    cfg = SCENARIOS[scenario]
    bars: int = cfg["bars"]
    drops = cfg["drops"]
    if drops is None:
        # sideways: 相对基准价 ±1% 随机偏移（不累积）
        rng = random.Random(seed)
        prices = [round(base_price * (1 + rng.uniform(-0.01, 0.01)), 2) for _ in range(bars)]
        return prices
    prices = []
    current = base_price
    for d in drops:
        current = current * (1 + d)
        prices.append(round(current, 2))
    return prices


def inject_adversarial(
    prices: list[float],
    scenario: ScenarioType,
    inject_at: int,
    seed: int | None = None,
) -> list[float]:
    """将对抗性价格序列注入到已有价格序列的指定位置。

    Args:
        prices: 原始价格序列
        scenario: 场景类型
        inject_at: 注入位置（index）
        seed: 随机种子

    Returns:
        修改后的价格序列（长度不变，原地替换 inject_at 起的连续 N 根）
    """
    adv_prices = generate_adversarial_prices(
        prices[inject_at] if inject_at < len(prices) else prices[-1],
        scenario, seed,
    )
    result = prices.copy()
    for i, ap in enumerate(adv_prices):
        pos = inject_at + i
        if pos < len(result):
            result[pos] = ap
    return result
