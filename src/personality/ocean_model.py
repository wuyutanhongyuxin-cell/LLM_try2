from __future__ import annotations

"""Big Five (OCEAN) 人格模型定义。

定义 OceanProfile 数据结构和 7 个预定义交易员人格原型。
每个维度 0-100 连续分数，决定 Agent 的交易风格和风控参数。
"""

from pydantic import BaseModel, Field


class OceanProfile(BaseModel):
    """Big Five人格参数，每个维度0-100"""

    name: str = Field(..., description="人格原型名称，如'冷静创新型'")
    openness: int = Field(
        ..., ge=0, le=100,
        description="开放性: 高=探索新策略新币种, 低=只做主流",
    )
    conscientiousness: int = Field(
        ..., ge=0, le=100,
        description="尽责性: 高=严格风控纪律, 低=冲动交易",
    )
    extraversion: int = Field(
        ..., ge=0, le=100,
        description="外向性: 高=追随市场情绪, 低=逆向独立判断",
    )
    agreeableness: int = Field(
        ..., ge=0, le=100,
        description="宜人性: 高=从众跟风, 低=对抗市场共识",
    )
    neuroticism: int = Field(
        ..., ge=0, le=100,
        description="神经质: 高=极度厌恶损失/频繁止损, 低=能扛回撤",
    )


# 7 个预定义人格原型，覆盖典型交易风格
PRESET_PROFILES: dict[str, OceanProfile] = {
    "冷静创新型": OceanProfile(
        name="冷静创新型", openness=90, conscientiousness=80,
        extraversion=25, agreeableness=20, neuroticism=10,
    ),
    "保守焦虑型": OceanProfile(
        name="保守焦虑型", openness=15, conscientiousness=85,
        extraversion=20, agreeableness=70, neuroticism=90,
    ),
    "激进冒险型": OceanProfile(
        name="激进冒险型", openness=85, conscientiousness=20,
        extraversion=80, agreeableness=15, neuroticism=10,
    ),
    "纪律动量型": OceanProfile(
        name="纪律动量型", openness=50, conscientiousness=90,
        extraversion=75, agreeableness=50, neuroticism=30,
    ),
    "逆向价值型": OceanProfile(
        name="逆向价值型", openness=60, conscientiousness=75,
        extraversion=10, agreeableness=10, neuroticism=25,
    ),
    "平衡中庸型": OceanProfile(
        name="平衡中庸型", openness=50, conscientiousness=50,
        extraversion=50, agreeableness=50, neuroticism=50,
    ),
    "情绪追涨型": OceanProfile(
        name="情绪追涨型", openness=70, conscientiousness=15,
        extraversion=90, agreeableness=80, neuroticism=75,
    ),
}


def get_profile(name: str) -> OceanProfile:
    """根据名称获取预定义人格原型，不存在则抛出 KeyError。"""
    if name not in PRESET_PROFILES:
        available = ", ".join(PRESET_PROFILES.keys())
        raise KeyError(f"未知人格原型 '{name}'，可用: {available}")
    return PRESET_PROFILES[name]
