from __future__ import annotations

"""核心交易 Agent：继承 BaseAgent，实现 LLM 决策循环。"""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal

from litellm import acompletion
from loguru import logger

from src.agent.base_agent import BaseAgent
from src.agent.memory import AgentMemory
from src.agent.multi_sample import vote_on_actions
from src.agent.reflection import generate_meta_reflection, generate_reflection
from src.execution.signal import Action, TradeSignal
from src.execution.strategy import RuleBasedStrategy
from src.integration.redis_bus import RedisBus
from src.market.data_feed import DataFeed, MarketSnapshot
from src.personality.ocean_model import OceanProfile
from src.personality.prompt_generator import (
    generate_decision_prompt, generate_system_prompt, get_prompt_hash,
)
from src.personality.trait_to_constraint import TradingConstraints
from src.utils.anonymizer import AssetAnonymizer

SIGNAL_CHANNEL: str = "agent_signals"


def _snapshot_to_dict(snapshot: MarketSnapshot) -> dict:
    """MarketSnapshot -> prompt_generator 所需的 dict。"""
    return {
        "asset": snapshot.asset, "price": snapshot.price,
        "change_24h": snapshot.price_24h_change_pct, "volume": snapshot.volume_24h,
    }


class TradingAgent(BaseAgent):
    """基于 OCEAN 人格的交易 Agent，LLM 驱动决策。"""

    def __init__(
        self, agent_id: str, profile: OceanProfile,
        constraints: TradingConstraints, llm_config: dict,
        market_feed: DataFeed, redis_bus: RedisBus,
        market_type: str = "crypto",
        leverage: int = 1, mmr: float = 0.004,
    ) -> None:
        """初始化交易 Agent。leverage>1 时自动注入杠杆风险约束。"""
        super().__init__(agent_id, profile.name)
        self._profile: OceanProfile = profile
        self._constraints: TradingConstraints = constraints
        self._llm_config: dict = llm_config
        self._market_feed: DataFeed = market_feed
        self._redis_bus: RedisBus = redis_bus
        self._market_type: str = market_type
        self._memory: AgentMemory = AgentMemory(agent_id, redis_bus)
        self._system_prompt: str = generate_system_prompt(
            profile, constraints, market_type, leverage=leverage, mmr=mmr,
        )
        self._prompt_hash: str = get_prompt_hash(self._system_prompt)
        self._positions: list[dict] = []
        self._portfolio_value: Decimal = Decimal("10000")
        self._trade_count: int = 0
        self._anonymizer: AssetAnonymizer | None = None
        self._strategy: RuleBasedStrategy = RuleBasedStrategy(
            agent_id=agent_id, agent_name=profile.name,
            profile_dump=profile.model_dump(exclude={"name"}),
            prompt_hash=self._prompt_hash,
            llm_model=llm_config.get("model", ""),
            leverage=leverage, mmr=mmr,
        )

    # ── 主循环 ──────────────────────────────────────────

    async def _run_loop(self) -> None:
        """按 rebalance 间隔循环执行决策。"""
        interval: int = self._constraints.rebalance_interval_seconds
        logger.info(f"[{self._name}] 决策间隔: {interval}秒")
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._decision_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[{self._name}] 决策循环异常: {exc}")

    # ── 单次决策 ────────────────────────────────────────

    async def _decision_cycle(self) -> None:
        """一次完整决策：行情→匿名化→Prompt→LLM(多采样)→反匿名→校验→发布。"""
        asset: str = self._constraints.allowed_assets[0]
        snapshot: MarketSnapshot | None = await self._market_feed.get_latest(asset)
        if snapshot is None:
            logger.warning(f"[{self._name}] 行情获取失败，跳过本轮")
            return
        # C1: 传入当前 asset 用于相关性检索
        context: str = await self._memory.get_context_for_decision(
            current_asset=asset, current_action="",
        )
        market_dict = _snapshot_to_dict(snapshot)
        # A3: 匿名化（如果启用）
        if self._anonymizer:
            market_dict = self._anonymizer.anonymize_market_data(market_dict)
            context = self._anonymizer.anonymize(context)
        prompt: str = generate_decision_prompt(
            market_dict, self._positions, context, float(self._portfolio_value),
        )
        # B2: 多采样投票
        n_samples: int = self._llm_config.get("decision_samples", 3)
        threshold: float = self._llm_config.get("consensus_threshold", 0.6)
        if n_samples <= 1:
            raw = await self._call_llm(prompt)
            if raw is None:
                return
            signal = self._validate_signal(raw, snapshot)
        else:
            signal = await self._multi_sample_decision(prompt, snapshot, n_samples, threshold)
        if signal is not None:
            await self._execute_signal(signal)

    async def _multi_sample_decision(
        self, prompt: str, snapshot: MarketSnapshot,
        n_samples: int, threshold: float,
    ) -> TradeSignal | None:
        """多次调用 LLM，投票决定最终信号。"""
        parsed: list[dict] = []
        for i in range(n_samples):
            raw = await self._call_llm(prompt)
            if raw is None:
                continue
            # A3: 反匿名化 LLM 响应中的资产名
            if self._anonymizer:
                raw = self._anonymizer.deanonymize(raw)
            data = self._parse_llm_response(raw)
            if data is not None:
                parsed.append(data)
        winner = vote_on_actions(parsed, consensus_threshold=threshold)
        if winner is None:
            logger.info(f"[{self._name}] 多采样无共识，默认 HOLD")
            return None
        # 用胜出的 data 构建信号
        return self._build_signal_from_data(winner, snapshot)

    # ── LLM 调用（含重试） ─────────────────────────────

    async def _call_llm(self, user_prompt: str) -> str | None:
        """调用 LLM，失败自动重试，全部失败返回 None。"""
        retry_count: int = self._llm_config.get("retry_count", 3)
        retry_delay: float = self._llm_config.get("retry_delay_seconds", 5.0)
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(1, retry_count + 1):
            try:
                resp = await acompletion(
                    model=self._llm_config["model"],
                    messages=messages,
                    temperature=self._llm_config.get("temperature", 0.3),
                    max_tokens=self._llm_config.get("max_tokens", 1024),
                    timeout=self._llm_config.get("timeout_seconds", 30),
                )
                return resp.choices[0].message.content  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning(f"[{self._name}] LLM 失败 ({attempt}/{retry_count}): {exc}")
                if attempt < retry_count:
                    await asyncio.sleep(retry_delay)
        logger.error(f"[{self._name}] LLM 调用全部失败，跳过本轮")
        return None

    # ── JSON 解析 ───────────────────────────────────────

    def _parse_llm_response(self, raw: str) -> dict | None:
        """解析 LLM 原始输出为 dict，失败返回 None。"""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"[{self._name}] JSON 解析失败，原始响应: {raw[:200]}")
            return None

    # ── 信号校验（硬约束由代码强制执行） ────────────────

    def _validate_signal(self, raw: str, snapshot: MarketSnapshot) -> TradeSignal | None:
        """校验 LLM 输出，clip 超限字段，返回合法 TradeSignal 或 None。"""
        data: dict | None = self._parse_llm_response(raw)
        if data is None:
            return None
        # A3: 反匿名化
        if self._anonymizer:
            asset_raw = str(data.get("asset", ""))
            data["asset"] = self._anonymizer.deanonymize_asset(asset_raw)
        return self._build_signal_from_data(data, snapshot)

    def _build_signal_from_data(self, data: dict, snapshot: MarketSnapshot) -> TradeSignal | None:
        """委托给执行策略处理 LLM 原始输出。"""
        return self._strategy.process_signal(
            data, snapshot, self._constraints, float(self._portfolio_value),
        )

    # ── 信号执行（发布 + 记忆更新） ────────────────────

    async def _execute_signal(self, signal: TradeSignal) -> None:
        """发布信号到 Redis，更新记忆层。"""
        await self._redis_bus.publish(SIGNAL_CHANNEL, signal.model_dump())
        logger.info(
            f"[{self._name}] 信号: {signal.action.value} {signal.asset} "
            f"size={signal.size_pct:.1f}% conf={signal.confidence:.2f}"
        )
        # 更新 L1 + L2 记忆
        trade_data = signal.model_dump()
        self._memory.add_trade_result(trade_data)
        await self._memory.save_trade_to_l2(trade_data)
        # 更新 L1 tick（记录行情快照）
        self._memory.add_tick({"price": signal.entry_price, "asset": signal.asset})
        self._trade_count += 1
        # 每 10 笔交易触发反思（更新 L3 语义记忆）
        if self._trade_count % 10 == 0:
            await self._trigger_reflection()

    # ── 反思（每 10 笔交易触发） ──────────────────────

    async def _trigger_reflection(self) -> None:
        """调用反思模块，结果存入 L3 记忆。每 30 笔额外触发元反思。"""
        recent = await self._memory.get_recent_trades(count=10)
        result = await generate_reflection(
            self._name, self._profile, recent, self._llm_config,
        )
        if result and "summary" in result:
            await self._memory.add_reflection(result["summary"])
            logger.info(f"[{self._name}] 第 {self._trade_count} 笔，反思完成")
        # 每 30 笔交易触发元反思
        if self._trade_count % 30 == 0:
            await self._trigger_meta_reflection()

    async def _trigger_meta_reflection(self) -> None:
        """调用元反思模块，对最近 3 条 L3 反思进行二阶反思。"""
        reflections = await self._memory._get_reflections(count=3)
        if len(reflections) < 3:
            return  # 反思不足 3 条，跳过
        result = await generate_meta_reflection(
            self._name, self._profile, reflections, self._llm_config,
        )
        if result and "meta_summary" in result:
            meta_entry = f"[META] {result['meta_summary']}"
            await self._memory.add_reflection(meta_entry)
            logger.info(f"[{self._name}] 第 {self._trade_count} 笔，元反思完成")
