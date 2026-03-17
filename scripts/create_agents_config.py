from __future__ import annotations

"""批量生成 Agent OCEAN 配置。

用法: python scripts/create_agents_config.py [数量] [输出文件]
默认: 生成 20 个随机 OCEAN 组合到 config/agents_generated.yaml

包含所有 7 个预定义原型 + 随机组合，输出为 YAML 格式。
"""

import random
import sys
from pathlib import Path

import yaml

from src.personality.ocean_model import PRESET_PROFILES

# 默认参数
DEFAULT_COUNT: int = 20
DEFAULT_OUTPUT: str = "config/agents_generated.yaml"
DEFAULT_CAPITAL: int = 10000


def _generate_preset_entries() -> list[dict]:
    """生成所有 7 个预定义原型的配置条目。"""
    entries: list[dict] = []
    for i, (name, profile) in enumerate(PRESET_PROFILES.items()):
        entry: dict = {
            "id": f"agent_preset_{i}",
            "preset": name,
            "initial_capital": DEFAULT_CAPITAL,
        }
        entries.append(entry)
    return entries


def _generate_random_entry(index: int) -> dict:
    """生成一个随机 OCEAN 组合的配置条目。

    Args:
        index: 序号，用于生成唯一 id

    Returns:
        YAML 配置字典
    """
    o = random.randint(0, 100)
    c = random.randint(0, 100)
    e = random.randint(0, 100)
    a = random.randint(0, 100)
    n = random.randint(0, 100)
    return {
        "id": f"agent_random_{index}",
        "custom": {
            "name": f"随机型_{index}",
            "openness": o,
            "conscientiousness": c,
            "extraversion": e,
            "agreeableness": a,
            "neuroticism": n,
        },
        "initial_capital": DEFAULT_CAPITAL,
    }


def generate_config(total_count: int) -> dict:
    """生成完整配置：7 个预定义 + N 个随机。

    Args:
        total_count: 期望的 Agent 总数（至少 7 个预定义）

    Returns:
        可直接写入 YAML 的字典
    """
    entries = _generate_preset_entries()
    random_count = max(0, total_count - len(entries))
    for i in range(random_count):
        entries.append(_generate_random_entry(i))
    return {"agents": entries}


def main() -> None:
    """解析命令行参数并生成配置文件。"""
    # 解析参数
    count = DEFAULT_COUNT
    output_path = DEFAULT_OUTPUT
    if len(sys.argv) >= 2:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"错误: 数量必须为整数，收到 '{sys.argv[1]}'")
            sys.exit(1)
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    # 生成配置
    config = generate_config(count)

    # 写入文件
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False,
        )

    preset_count = len(PRESET_PROFILES)
    random_count = len(config["agents"]) - preset_count
    print(f"已生成 {len(config['agents'])} 个 Agent 配置:")
    print(f"  - 预定义原型: {preset_count} 个")
    print(f"  - 随机组合:   {random_count} 个")
    print(f"  - 输出文件:   {out.resolve()}")


if __name__ == "__main__":
    main()
