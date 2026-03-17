from __future__ import annotations
"""全链路交易日志记录器。

每笔交易记录完整决策链：行情→Prompt hash→LLM响应→校验对比→执行结果→成本明细。
存储到 Redis list，key = trade_log:{agent_id}
"""
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.integration.redis_bus import RedisBus

_LOG_LIMIT: int = 200  # 每个 Agent 最多保留 200 条日志


class TradeLogEntry(BaseModel):
    """单笔交易的完整链路日志。"""

    agent_id: str = Field(..., description="Agent 标识")
    timestamp: str = Field(..., description="时间戳")
    market_snapshot: dict = Field(default_factory=dict, description="原始行情数据")
    prompt_hash: str = Field(default="", description="System + User prompt 的 SHA256 前12位")
    llm_model: str = Field(default="", description="LLM 模型标识")
    llm_raw_response: str = Field(
        default="", max_length=500, description="LLM 原始响应（截取前500字符）"
    )
    pre_clip_signal: dict = Field(default_factory=dict, description="校验前的 LLM 原始输出")
    post_clip_signal: dict = Field(default_factory=dict, description="校验后的最终信号")
    fields_clipped: list[str] = Field(default_factory=list, description="被 clip 的字段名列表")
    executed: bool = Field(default=False, description="是否最终执行")
    execution_result: dict = Field(default_factory=dict, description="执行结果")
    cost_breakdown: dict = Field(default_factory=dict, description="成本明细")


class TradeLogger:
    """交易日志记录器，写入 Redis。"""

    def __init__(self, redis_bus: RedisBus) -> None:
        self._redis = redis_bus

    def _key(self, agent_id: str) -> str:
        """生成 Redis key。"""
        return f"trade_log:{agent_id}"

    async def log_trade(self, entry: TradeLogEntry) -> None:
        """记录一条交易日志。"""
        try:
            await self._redis.lpush_json(self._key(entry.agent_id), entry.model_dump())
            await self._redis.ltrim(self._key(entry.agent_id), 0, _LOG_LIMIT - 1)
        except Exception as exc:
            logger.error("交易日志写入失败 [{}]: {}", entry.agent_id, exc)

    async def get_agent_log(self, agent_id: str, count: int = 20) -> list[dict]:
        """获取指定 Agent 最近 N 条日志。"""
        return await self._redis.lrange_json(self._key(agent_id), 0, count - 1)
