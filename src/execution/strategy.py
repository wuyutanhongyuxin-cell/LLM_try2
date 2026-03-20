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
                 profile_dump: dict, prompt_hash: str, llm_model: str,
                 leverage: int = 1, mmr: float = 0.004) -> None:
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._profile_dump = profile_dump
        self._prompt_hash = prompt_hash
        self._llm_model = llm_model
        self._leverage = leverage
        # 杠杆>1 时计算安全 SL 上限: (1/leverage - MMR) × safety_factor
        if leverage > 1:
            liq_dist = (1.0 / leverage) - mmr
            self._max_sl_pct = liq_dist * 0.6 * 100  # 60% 安全系数，转百分比
        else:
            self._max_sl_pct = 100.0  # 无杠杆不限制

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
        if confidence < constraints.min_confidence_threshold:
            logger.info(f"[{self._agent_name}] 信心不足 {confidence:.2f}，跳过")
            return None
        stop_loss: float | None = raw_data.get("stop_loss_price")
        if constraints.require_stop_loss and stop_loss is None:
            logger.warning(f"[{self._agent_name}] 缺少止损价格，约束要求必须设置")
            return None
        entry = float(raw_data.get("entry_price", snapshot.price))
        take_profit: float | None = raw_data.get("take_profit_price")
        # 杠杆感知 SL/TP clip：防止爆仓
        if self._leverage > 1 and entry > 0:
            stop_loss, take_profit = self._clip_sl_tp(
                action_str, entry, stop_loss, take_profit,
            )
        return TradeSignal(
            agent_id=self._agent_id, agent_name=self._agent_name,
            timestamp=datetime.now(tz=timezone.utc),
            action=Action(action_str), asset=asset, size_pct=size_pct,
            entry_price=entry,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            confidence=confidence,
            reasoning=str(raw_data.get("reasoning", "")),
            personality_influence=str(raw_data.get("personality_influence", "")),
            ocean_profile=self._profile_dump,
            prompt_hash=self._prompt_hash,
            llm_model=self._llm_model,
        )

    def _clip_sl_tp(
        self, action: str, entry: float, sl: float | None, tp: float | None,
    ) -> tuple[float | None, float | None]:
        """杠杆感知 SL/TP clip：确保 SL 距离不超过爆仓安全阈值。"""
        max_dist = entry * self._max_sl_pct / 100.0
        if sl is not None:
            dist = abs(entry - sl)
            if dist > max_dist:
                old_sl = sl
                # BUY: SL 在下方; SELL: SL 在上方
                sl = entry - max_dist if action == "BUY" else entry + max_dist
                logger.warning(
                    f"[{self._agent_name}] SL 距离 {dist/entry*100:.1f}% 超限 "
                    f"({self._max_sl_pct:.1f}%), clip {old_sl:.0f}→{sl:.0f}")
        if tp is not None:
            # TP 限制为 SL 距离的 3 倍（最大 R:R 3:1）
            max_tp_dist = max_dist * 3
            tp_dist = abs(entry - tp)
            if tp_dist > max_tp_dist:
                old_tp = tp
                tp = entry + max_tp_dist if action == "BUY" else entry - max_tp_dist
                logger.info(f"[{self._agent_name}] TP clip {old_tp:.0f}→{tp:.0f}")
        return sl, tp
