from __future__ import annotations

"""交易信号数据结构定义。

Agent 决策循环产出 TradeSignal，经校验后发布到 Redis 消息总线。
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Action(str, Enum):
    """交易动作枚举。"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeSignal(BaseModel):
    """交易信号，由 Agent 决策生成。

    所有金额/百分比字段由代码根据 TradingConstraints 校验并 clip，
    LLM 不可直接控制最终执行值。
    """

    agent_id: str = Field(..., description="Agent 唯一标识")
    agent_name: str = Field(..., description="人格原型名称")
    timestamp: datetime = Field(..., description="信号生成时间")
    action: Action = Field(..., description="交易动作: BUY/SELL/HOLD")
    asset: str = Field(..., description="交易资产标识，如 BTC-PERP")
    size_pct: float = Field(..., description="占总资产百分比，由 constraints 限制上限")
    entry_price: float = Field(..., description="入场价格")
    stop_loss_price: Optional[float] = Field(None, description="止损价格")
    take_profit_price: Optional[float] = Field(None, description="止盈价格")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="决策信心度 0.0-1.0"
    )
    reasoning: str = Field(..., description="决策理由（英文，LLM 原始输出）")
    personality_influence: str = Field(..., description="主导的 OCEAN 维度说明")
    ocean_profile: dict = Field(..., description="完整 OCEAN 五维分数快照")
    prompt_hash: str = Field(default="", description="生成此信号时使用的 prompt 版本 hash")
    llm_model: str = Field(default="", description="LLM 模型版本标识")
