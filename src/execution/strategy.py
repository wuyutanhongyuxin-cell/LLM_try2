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


def _safe_float(value: object, default: float = 0.0) -> float:
    """安全转 float，nan/inf/异常均返回 default。"""
    try:
        v = float(value)  # type: ignore[arg-type]
        if v != v or v == float("inf") or v == float("-inf"):  # nan/inf
            return default
        return v
    except (ValueError, TypeError):
        return default


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
            if liq_dist <= 0:
                # 杠杆过高或 MMR 配置错误，用最小安全距离兜底
                logger.error(
                    f"[{agent_name}] liq_dist={liq_dist:.6f} <= 0 "
                    f"(leverage={leverage}, mmr={mmr})，使用最小安全距离 0.1%")
                liq_dist = 0.001  # 0.1% 最小兜底
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
        # Bug1 修复：HOLD = 无动作，直接返回 None，不走后续 clip 流程
        if action_str == "HOLD":
            return None
        asset: str = str(raw_data.get("asset", ""))
        if asset not in constraints.allowed_assets:
            logger.warning(f"[{self._agent_name}] 资产 {asset} 不在允许列表中")
            return None
        size_pct = _clip(_safe_float(raw_data.get("size_pct")), 0, constraints.max_position_pct)
        confidence = _clip(_safe_float(raw_data.get("confidence")), 0.0, 1.0)
        if confidence < constraints.min_confidence_threshold:
            logger.info(f"[{self._agent_name}] 信心不足 {confidence:.2f}，跳过")
            return None
        # SL/TP 为 0 或非法值视为"未设置"（None）
        stop_loss_raw = raw_data.get("stop_loss_price")
        stop_loss: float | None = _safe_float(stop_loss_raw) if stop_loss_raw is not None else None
        if stop_loss is not None and stop_loss == 0:
            stop_loss = None
        if constraints.require_stop_loss and stop_loss is None:
            logger.warning(f"[{self._agent_name}] 缺少止损价格，约束要求必须设置")
            return None
        entry = _safe_float(raw_data.get("entry_price", snapshot.price))
        if entry <= 0:
            entry = float(snapshot.price)
        tp_raw = raw_data.get("take_profit_price")
        take_profit: float | None = _safe_float(tp_raw) if tp_raw is not None else None
        if take_profit is not None and take_profit == 0:
            take_profit = None
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
        """杠杆感知 SL/TP clip：方向校验 + 距离限制。

        Bug5 修复：先检查 SL/TP 方向是否正确（BUY 的 SL 必须在下方，
        SELL 的 SL 必须在上方），方向错误则强制修正。
        参考 Binance API 规则：方向错误直接拒单。
        """
        max_dist = entry * self._max_sl_pct / 100.0
        is_buy = action == "BUY"

        if sl is not None:
            sl = float(sl)
            # Bug5：方向校验（BUY SL 必须 < entry，SELL SL 必须 > entry）
            direction_wrong = (is_buy and sl >= entry) or (not is_buy and sl <= entry)
            if direction_wrong:
                old_sl = sl
                sl = entry - max_dist if is_buy else entry + max_dist
                logger.warning(
                    f"[{self._agent_name}] SL 方向错误: "
                    f"{'BUY' if is_buy else 'SELL'} SL={old_sl:.0f} vs "
                    f"entry={entry:.0f}, 修正→{sl:.0f}")
            else:
                # 距离校验
                dist = abs(entry - sl)
                if dist > max_dist:
                    old_sl = sl
                    sl = entry - max_dist if is_buy else entry + max_dist
                    logger.warning(
                        f"[{self._agent_name}] SL 距离 {dist/entry*100:.1f}% 超限 "
                        f"({self._max_sl_pct:.1f}%), clip {old_sl:.0f}→{sl:.0f}")

        if tp is not None:
            tp = float(tp)
            # TP 方向校验（BUY TP 必须 > entry，SELL TP 必须 < entry）
            tp_dir_wrong = (is_buy and tp <= entry) or (not is_buy and tp >= entry)
            if tp_dir_wrong:
                max_tp_dist = max_dist * 3
                old_tp = tp
                tp = entry + max_tp_dist if is_buy else entry - max_tp_dist
                logger.warning(
                    f"[{self._agent_name}] TP 方向错误: "
                    f"{'BUY' if is_buy else 'SELL'} TP={old_tp:.0f} vs "
                    f"entry={entry:.0f}, 修正→{tp:.0f}")
            else:
                # TP 限制为 SL 距离的 3 倍（最大 R:R 3:1）
                max_tp_dist = max_dist * 3
                tp_dist = abs(entry - tp)
                if tp_dist > max_tp_dist:
                    old_tp = tp
                    tp = entry + max_tp_dist if is_buy else entry - max_tp_dist
                    logger.info(
                        f"[{self._agent_name}] TP clip {old_tp:.0f}→{tp:.0f}")

        return sl, tp
