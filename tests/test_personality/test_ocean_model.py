from __future__ import annotations

"""OceanProfile 模型与 32 个预定义原型测试。"""

import pytest
from pydantic import ValidationError

from src.personality.ocean_model import (
    PRESET_PROFILES,
    OceanProfile,
    get_profile,
)


# ── 预定义原型参数正确性 ────────────────────────────────

class TestPresetProfiles:
    """验证预定义原型的 OCEAN 参数值。"""

    def test_preset_count(self) -> None:
        """必须恰好 32 个预定义原型（2^5 二元组合）。"""
        assert len(PRESET_PROFILES) == 32

    # 4 个经典原型：保留原始参数，确保向后兼容
    def test_calm_innovator(self) -> None:
        p = PRESET_PROFILES["冷静创新型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (90, 80, 25, 20, 10)

    def test_conservative_anxious(self) -> None:
        p = PRESET_PROFILES["保守焦虑型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (15, 85, 20, 70, 90)

    def test_aggressive_risk_taker(self) -> None:
        p = PRESET_PROFILES["激进冒险型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (85, 20, 80, 15, 10)

    def test_emotional_chaser(self) -> None:
        p = PRESET_PROFILES["情绪追涨型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (70, 15, 90, 80, 75)

    # 新增原型抽样验证（每象限各 1 个）
    def test_lazy_headwind(self) -> None:
        """O↓C↓ 保守散漫系：散漫逆风型 = LLLLL"""
        p = PRESET_PROFILES["散漫逆风型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (20, 20, 20, 20, 20)

    def test_iron_defense(self) -> None:
        """O↓C↑ 保守纪律系：铁壁防守型 = LHLLL"""
        p = PRESET_PROFILES["铁壁防守型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (20, 80, 20, 20, 20)

    def test_wild_hunter(self) -> None:
        """O↑C↓ 探索冲动系：狂野猎手型 = HLLLL"""
        p = PRESET_PROFILES["狂野猎手型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (80, 20, 20, 20, 20)

    def test_all_round_dominant(self) -> None:
        """O↑C↑ 探索纪律系：全能主导型 = HHHLL"""
        p = PRESET_PROFILES["全能主导型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (80, 80, 80, 20, 20)

    def test_all_high(self) -> None:
        """极端值：全面紧绷型 = HHHHH"""
        p = PRESET_PROFILES["全面紧绷型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (80, 80, 80, 80, 80)


class TestBinaryCoverage:
    """验证 32 个原型完整覆盖 2^5 二元空间。"""

    def test_all_32_binary_codes_covered(self) -> None:
        """每个 H/L 二元组合都有且只有一个对应原型。"""
        codes: set[str] = set()
        for p in PRESET_PROFILES.values():
            code = (
                ("H" if p.openness > 50 else "L")
                + ("H" if p.conscientiousness > 50 else "L")
                + ("H" if p.extraversion > 50 else "L")
                + ("H" if p.agreeableness > 50 else "L")
                + ("H" if p.neuroticism > 50 else "L")
            )
            codes.add(code)
        assert len(codes) == 32

    def test_no_duplicate_binary_codes(self) -> None:
        """不能有两个原型映射到同一个二元组合。"""
        codes: list[str] = []
        for p in PRESET_PROFILES.values():
            code = (
                ("H" if p.openness > 50 else "L")
                + ("H" if p.conscientiousness > 50 else "L")
                + ("H" if p.extraversion > 50 else "L")
                + ("H" if p.agreeableness > 50 else "L")
                + ("H" if p.neuroticism > 50 else "L")
            )
            codes.append(code)
        assert len(codes) == len(set(codes))


# ── 字段校验 ─────────────────────────────────────────────

class TestOceanProfileValidation:
    """测试 Pydantic 校验规则。"""

    def test_valid_boundary_zero(self) -> None:
        p = OceanProfile(name="min", openness=0, conscientiousness=0,
                         extraversion=0, agreeableness=0, neuroticism=0)
        assert p.openness == 0

    def test_valid_boundary_hundred(self) -> None:
        p = OceanProfile(name="max", openness=100, conscientiousness=100,
                         extraversion=100, agreeableness=100, neuroticism=100)
        assert p.neuroticism == 100

    def test_reject_negative(self) -> None:
        with pytest.raises(ValidationError):
            OceanProfile(name="bad", openness=-1, conscientiousness=50,
                         extraversion=50, agreeableness=50, neuroticism=50)

    def test_reject_over_hundred(self) -> None:
        with pytest.raises(ValidationError):
            OceanProfile(name="bad", openness=101, conscientiousness=50,
                         extraversion=50, agreeableness=50, neuroticism=50)

    def test_reject_negative_neuroticism(self) -> None:
        with pytest.raises(ValidationError):
            OceanProfile(name="bad", openness=50, conscientiousness=50,
                         extraversion=50, agreeableness=50, neuroticism=-10)

    def test_reject_over_hundred_conscientiousness(self) -> None:
        with pytest.raises(ValidationError):
            OceanProfile(name="bad", openness=50, conscientiousness=200,
                         extraversion=50, agreeableness=50, neuroticism=50)

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            OceanProfile(openness=50, conscientiousness=50,
                         extraversion=50, agreeableness=50, neuroticism=50)  # type: ignore[call-arg]


# ── get_profile 函数 ─────────────────────────────────────

class TestGetProfile:
    """测试预定义原型查找函数。"""

    def test_get_existing_profile(self) -> None:
        p = get_profile("冷静创新型")
        assert p.name == "冷静创新型"
        assert p.openness == 90

    def test_get_all_presets(self) -> None:
        """所有 32 个原型都能通过 get_profile 获取。"""
        for name in PRESET_PROFILES:
            p = get_profile(name)
            assert p.name == name

    def test_unknown_profile_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="未知人格原型"):
            get_profile("不存在的类型")

    def test_removed_profiles_raise_error(self) -> None:
        """旧的 3 个中间值原型已被替换为二元类型。"""
        for old_name in ("纪律动量型", "逆向价值型", "平衡中庸型"):
            with pytest.raises(KeyError):
                get_profile(old_name)
