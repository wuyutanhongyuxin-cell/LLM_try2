"""OCEAN 人格参数 → 交易硬约束映射。公式固定，不可被 LLM 覆盖。"""

from pydantic import BaseModel, Field

from src.personality.ocean_model import OceanProfile


class TradingConstraints(BaseModel):
    """由 OCEAN 人格参数计算出的交易硬约束，代码强制执行。"""

    max_position_pct: float = Field(..., description="单笔最大仓位占总资金百分比")
    stop_loss_pct: float = Field(..., description="止损百分比")
    max_drawdown_pct: float = Field(..., description="最大回撤容忍度")
    max_concurrent_positions: int = Field(..., description="最大同时持仓数")
    rebalance_interval_seconds: int = Field(..., description="再平衡检查间隔(秒)")
    allowed_assets: list[str] = Field(..., description="允许交易的资产列表")
    use_sentiment_data: bool = Field(..., description="是否使用情绪数据")
    momentum_weight: float = Field(..., description="动量信号权重 0.0-1.0")
    contrarian_weight: float = Field(..., description="逆向信号权重 0.0-1.0")
    require_stop_loss: bool = Field(..., description="是否强制要求止损")
    min_confidence_threshold: float = Field(..., description="最低信心阈值 0.0-1.0")


def _clip(value: float, min_val: float, max_val: float) -> float:
    """将 value 限制在 [min_val, max_val] 范围内。"""
    return max(min_val, min(value, max_val))


def ocean_to_constraints(
    profile: OceanProfile, global_config: dict
) -> TradingConstraints:
    """将 OCEAN 人格参数转换为交易硬约束。

    Args:
        profile: OCEAN 人格配置（五维度各 0-100）
        global_config: 需含 "major_assets" 和 "all_assets" 两个列表
    """
    o, c, e, a, n = (
        profile.openness, profile.conscientiousness,
        profile.extraversion, profile.agreeableness, profile.neuroticism,
    )
    # 神经质(N)越高 → 仓位越小、止损越紧、回撤容忍越低
    max_position_pct = _clip(5 + (100 - n) * 0.25, 5, 30)
    stop_loss_pct = _clip(1 + (100 - n) * 0.14, 1, 15)
    max_drawdown_pct = _clip(2 + (100 - n) * 0.18, 2, 20)
    # 开放性(O)越高 → 同时持仓越多、可交易资产越广
    max_concurrent_positions = int(_clip(1 + o // 20, 1, 6))
    allowed_assets = (
        global_config["all_assets"] if o > 60 else global_config["major_assets"]
    )
    # 神经质决定再平衡频率：越焦虑检查越频繁
    if n > 70:
        rebalance = 300        # 5分钟
    elif n > 40:
        rebalance = 3600       # 1小时
    else:
        rebalance = 86400      # 1天
    # 外向性(E) → 情绪数据 + 动量权重；宜人性(A)低 → 逆向权重高
    use_sentiment_data = e > 50
    momentum_weight = e / 100.0
    contrarian_weight = (100 - a) / 100.0
    # 尽责性(C) → 强制止损 + 最低信心阈值
    require_stop_loss = c > 50
    min_confidence_threshold = _clip(c / 100.0 * 0.8, 0.2, 0.8)

    return TradingConstraints(
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_concurrent_positions=max_concurrent_positions,
        rebalance_interval_seconds=rebalance,
        allowed_assets=allowed_assets,
        use_sentiment_data=use_sentiment_data,
        momentum_weight=momentum_weight,
        contrarian_weight=contrarian_weight,
        require_stop_loss=require_stop_loss,
        min_confidence_threshold=min_confidence_threshold,
    )
