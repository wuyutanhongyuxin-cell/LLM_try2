from __future__ import annotations

"""OceanProfile 模型与预定义原型测试。"""

import pytest
from pydantic import ValidationError

from src.personality.ocean_model import (
    PRESET_PROFILES,
    OceanProfile,
    get_profile,
)


# ── 预定义原型参数正确性 ────────────────────────────────

class TestPresetProfiles:
    """验证 7 个预定义原型的 OCEAN 参数值。"""

    def test_calm_innovator(self) -> None:
        p = PRESET_PROFILES["冷静创新型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (90, 80, 25, 20, 10)

    def test_conservative_anxious(self) -> None:
        p = PRESET_PROFILES["保守焦虑型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (15, 85, 20, 70, 90)

    def test_aggressive_risk_taker(self) -> None:
        p = PRESET_PROFILES["激进冒险型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (85, 20, 80, 15, 10)

    def test_disciplined_momentum(self) -> None:
        p = PRESET_PROFILES["纪律动量型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (50, 90, 75, 50, 30)

    def test_contrarian_value(self) -> None:
        p = PRESET_PROFILES["逆向价值型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (60, 75, 10, 10, 25)

    def test_balanced(self) -> None:
        p = PRESET_PROFILES["平衡中庸型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (50, 50, 50, 50, 50)

    def test_emotional_chaser(self) -> None:
        p = PRESET_PROFILES["情绪追涨型"]
        assert (p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism) == (70, 15, 90, 80, 75)

    def test_preset_count(self) -> None:
        """必须恰好 7 个预定义原型。"""
        assert len(PRESET_PROFILES) == 7


# ── 字段校验 ─────────────────────────────────────────────

class TestOceanProfileValidation:
    """测试 Pydantic 校验规则。"""

    def test_valid_boundary_zero(self) -> None:
        """0 是合法最小值。"""
        p = OceanProfile(name="min", openness=0, conscientiousness=0,
                         extraversion=0, agreeableness=0, neuroticism=0)
        assert p.openness == 0

    def test_valid_boundary_hundred(self) -> None:
        """100 是合法最大值。"""
        p = OceanProfile(name="max", openness=100, conscientiousness=100,
                         extraversion=100, agreeableness=100, neuroticism=100)
        assert p.neuroticism == 100

    def test_reject_negative(self) -> None:
        """负数应被 Pydantic 拒绝。"""
        with pytest.raises(ValidationError):
            OceanProfile(name="bad", openness=-1, conscientiousness=50,
                         extraversion=50, agreeableness=50, neuroticism=50)

    def test_reject_over_hundred(self) -> None:
        """超过 100 应被拒绝。"""
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
        """name 是必填字段。"""
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
        """所有 7 个原型都能通过 get_profile 获取。"""
        for name in PRESET_PROFILES:
            p = get_profile(name)
            assert p.name == name

    def test_unknown_profile_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="未知人格原型"):
            get_profile("不存在的类型")
