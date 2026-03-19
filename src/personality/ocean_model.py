from __future__ import annotations

"""Big Five (OCEAN) 人格模型定义。

基于 2^5 二元组合理论（SLOAN 分类体系），每个 OCEAN 维度分为 High/Low 两级，
穷举全部 32 种独特交易人格原型。4 个经典原型保留原始参数值以确保向后兼容。
新增原型统一使用 H=80, L=20 作为基准分数。
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


def _p(name: str, o: int, c: int, e: int, a: int, n: int) -> OceanProfile:
    """快捷构造器，减少样板代码。"""
    return OceanProfile(
        name=name, openness=o, conscientiousness=c,
        extraversion=e, agreeableness=a, neuroticism=n,
    )


# ═══════════════════════════════════════════════════════════════
# 32 个预定义人格原型（2^5 二元组合）
# 编码: O=开放性 C=尽责性 E=外向性 A=宜人性 N=神经质
# H=High(≥75) L=Low(≤25)  ★=经典原型（保留原始参数）
# ═══════════════════════════════════════════════════════════════
PRESET_PROFILES: dict[str, OceanProfile] = {
    # ── 4 个经典原型（原始参数，向后兼容，排在最前以匹配 --agents N） ──
    "冷静创新型": _p("冷静创新型", 90, 80, 25, 20, 10),   # ★ HHLLL
    "保守焦虑型": _p("保守焦虑型", 15, 85, 20, 70, 90),   # ★ LHLHH
    "激进冒险型": _p("激进冒险型", 85, 20, 80, 15, 10),   # ★ HLHLL
    "情绪追涨型": _p("情绪追涨型", 70, 15, 90, 80, 75),   # ★ HLHHH
    # ── O↓C↓ 保守散漫系 (8型) ──────────────────────────────
    "散漫逆风型": _p("散漫逆风型", 20, 20, 20, 20, 20),   # LLLLL
    "焦虑叛逆型": _p("焦虑叛逆型", 20, 20, 20, 20, 80),   # LLLLH
    "随性观望型": _p("随性观望型", 20, 20, 20, 80, 20),   # LLLHL
    "优柔寡断型": _p("优柔寡断型", 20, 20, 20, 80, 80),   # LLLHH
    "赌徒冲锋型": _p("赌徒冲锋型", 20, 20, 80, 20, 20),   # LLHLL
    "神经短线型": _p("神经短线型", 20, 20, 80, 20, 80),   # LLHLH
    "跟风散户型": _p("跟风散户型", 20, 20, 80, 80, 20),   # LLHHL
    "恐慌跟风型": _p("恐慌跟风型", 20, 20, 80, 80, 80),   # LLHHH
    # ── O↓C↑ 保守纪律系 (7型，保守焦虑型已列) ──────────────
    "铁壁防守型": _p("铁壁防守型", 20, 80, 20, 20, 20),   # LHLLL
    "谨慎狙击型": _p("谨慎狙击型", 20, 80, 20, 20, 80),   # LHLLH
    "稳健保守型": _p("稳健保守型", 20, 80, 20, 80, 20),   # LHLHL
    # LHLHH = 保守焦虑型（已在经典区）
    "纪律突击型": _p("纪律突击型", 20, 80, 80, 20, 20),   # LHHLL
    "精算套利型": _p("精算套利型", 20, 80, 80, 20, 80),   # LHHLH
    "纪律跟随型": _p("纪律跟随型", 20, 80, 80, 80, 20),   # LHHHL
    "风控趋势型": _p("风控趋势型", 20, 80, 80, 80, 80),   # LHHHH
    # ── O↑C↓ 探索冲动系 (6型，激进冒险型/情绪追涨型已列) ──
    "狂野猎手型": _p("狂野猎手型", 80, 20, 20, 20, 20),   # HLLLL
    "偏执创新型": _p("偏执创新型", 80, 20, 20, 20, 80),   # HLLLH
    "佛系探索型": _p("佛系探索型", 80, 20, 20, 80, 20),   # HLLHL
    "敏感探路型": _p("敏感探路型", 80, 20, 20, 80, 80),   # HLLHH
    # HLHLL = 激进冒险型（已在经典区）
    "躁动投机型": _p("躁动投机型", 80, 20, 80, 20, 80),   # HLHLH
    "乐观冲浪型": _p("乐观冲浪型", 80, 20, 80, 80, 20),   # HLHHL
    # HLHHH = 情绪追涨型（已在经典区）
    # ── O↑C↑ 探索纪律系 (7型，冷静创新型已列) ──────────────
    # HHLLL = 冷静创新型（已在经典区）
    "精密逆向型": _p("精密逆向型", 80, 80, 20, 20, 80),   # HHLLH
    "沉稳研究型": _p("沉稳研究型", 80, 80, 20, 80, 20),   # HHLHL
    "审慎观察型": _p("审慎观察型", 80, 80, 20, 80, 80),   # HHLHH
    "全能主导型": _p("全能主导型", 80, 80, 80, 20, 20),   # HHHLL
    "高压精英型": _p("高压精英型", 80, 80, 80, 20, 80),   # HHHLH
    "完美趋势型": _p("完美趋势型", 80, 80, 80, 80, 20),   # HHHHL
    "全面紧绷型": _p("全面紧绷型", 80, 80, 80, 80, 80),   # HHHHH
}


def get_profile(name: str) -> OceanProfile:
    """根据名称获取预定义人格原型，不存在则抛出 KeyError。"""
    if name not in PRESET_PROFILES:
        available = ", ".join(PRESET_PROFILES.keys())
        raise KeyError(f"未知人格原型 '{name}'，可用: {available}")
    return PRESET_PROFILES[name]
