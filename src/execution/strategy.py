"""执行策略抽象层。

定义 ExecutionStrategy 接口，当前提供 RuleBasedStrategy（即现有 clip 逻辑）。
未来 Phase 2/3 可实现 RLStrategy 替换，不改动 Agent 核心代码。

设计参考：LLM-guided RL (arxiv 2508.02366) 的信号层/执行层分离思想。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from loguru import logger

from src.execution.signal import Action, TradeSignal
from src.market.data_feed import MarketSnapshot
from src.personality.trait_to_constraint import TradingConstraints


def _clip(value: float, min_val: float, max_val: float) -> float:
    """将 value 限制在 [min_val, max_val] 范围内。"""
    return max(min_val, min(value, max_val))


class ExecutionStrategy(ABC):
    """执行策略抽象基类。接收 LLM 原始信号，输出最终可执行信号。"""

    @abstractmethod
    def process_signal(
        self,
        raw_data: dict,
        snapshot: MarketSnapshot,
        constraints: TradingConstraints,
        portfolio_value: float,
    ) -> TradeSignal | None:
        """处理 LLM 原始输出，返回最终可执行信号或 None。

        Args:
            raw_data: LLM JSON 解析后的 dict
            snapshot: 当前行情
            constraints: OCEAN 导出的交易约束
            portfolio_value: 当前总资产

        Returns:
            合法的 TradeSignal 或 None（拒绝执行）
        """
        ...


class RuleBasedStrategy(ExecutionStrategy):
    """基于规则的执行策略（当前默认）。

    将 trading_agent.py 中 _build_signal_from_data 的 clip 逻辑
    抽取到此处，使 Agent 核心代码不再直接包含校验逻辑。
    """

    def __init__(self, agent_id: str, agent_name: str,
                 profile_dump: dict, prompt_hash: str, llm_model: str) -> None:
        """初始化规则策略。"""
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._profile_dump = profile_dump
        self._prompt_hash = prompt_hash
        self._llm_model = llm_model

    def process_signal(
        self,
        raw_data: dict,
        snapshot: MarketSnapshot,
        constraints: TradingConstraints,
        portfolio_value: float,
    ) -> TradeSignal | None:
        """规则策略：clip + 白名单 + 信心阈值检查。"""
        action_str: str = str(raw_data.get("action", "")).upper()
        if action_str not in ("BUY", "SELL", "HOLD"):
            logger.warning(f"[{self._agent_name}] 无效 action: {action_str}")
            return None
        asset: str = str(raw_data.get("asset", ""))
        if asset not in constraints.allowed_assets:
            logger.warning(f"[{self._agent_name}] 资产 {asset} 不在允许列表中")
            return None
        size_pct = _clip(float(raw_data.get("size_pct", 0)), 0, constraints.max_position_pct)
        confidence = _clip(float(raw_data.get("confidence", 0)), 0.0, 1.0)
        # DeepSeek 兜底：BUY/SELL 返回 confidence=0.0 时，提升到 0.3（低信心但可交易）
        if action_str in ("BUY", "SELL") and confidence < 0.01:
            confidence = 0.3
            logger.info(f"[{self._agent_name}] confidence=0 兜底 → 0.3")
        if confidence < constraints.min_confidence_threshold:
            logger.info(f"[{self._agent_name}] 信心不足 {confidence:.2f}，跳过")
            return None
        stop_loss: float | None = raw_data.get("stop_loss_price")
        if constraints.require_stop_loss and stop_loss is None:
            logger.warning(f"[{self._agent_name}] 缺少止损价格，约束要求必须设置")
            return None
        return TradeSignal(
            agent_id=self._agent_id, agent_name=self._agent_name,
            timestamp=datetime.now(tz=timezone.utc),
            action=Action(action_str), asset=asset, size_pct=size_pct,
            entry_price=float(raw_data.get("entry_price", snapshot.price)),
            stop_loss_price=stop_loss,
            take_profit_price=raw_data.get("take_profit_price"),
            confidence=confidence,
            reasoning=str(raw_data.get("reasoning", "")),
            personality_influence=str(raw_data.get("personality_influence", "")),
            ocean_profile=self._profile_dump,
            prompt_hash=self._prompt_hash,
            llm_model=self._llm_model,
        )
