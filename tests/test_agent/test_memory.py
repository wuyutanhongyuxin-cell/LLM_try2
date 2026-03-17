from __future__ import annotations

"""三层记忆系统测试。L2/L3 使用 FakeRedisBus mock。"""

import pytest

from src.agent.memory import AgentMemory
from tests.conftest import FakeRedisBus


@pytest.fixture()
def memory() -> AgentMemory:
    """创建使用 FakeRedisBus 的 AgentMemory 实例。"""
    bus = FakeRedisBus()
    return AgentMemory(agent_id="test_agent", redis_bus=bus)  # type: ignore[arg-type]


# ── L1 Tick FIFO ────────────────────────────────────────

class TestL1TickFifo:
    """Working Memory tick 层：最多保留 20 条。"""

    def test_add_within_limit(self, memory: AgentMemory) -> None:
        """添加 10 条，全部保留。"""
        for i in range(10):
            memory.add_tick({"price": 100 + i})
        assert len(memory._working_ticks) == 10

    def test_fifo_overflow(self, memory: AgentMemory) -> None:
        """添加 25 条，只保留最新 20 条。"""
        for i in range(25):
            memory.add_tick({"price": 1000 + i})
        assert len(memory._working_ticks) == 20
        # 最老的应该是第 5 条（index=5），price=1005
        assert memory._working_ticks[0]["price"] == 1005
        # 最新的是第 24 条，price=1024
        assert memory._working_ticks[-1]["price"] == 1024


# ── L1 Trade FIFO ───────────────────────────────────────

class TestL1TradeFifo:
    """Working Memory trade 层：最多保留 5 条。"""

    def test_add_within_limit(self, memory: AgentMemory) -> None:
        for i in range(3):
            memory.add_trade_result({"action": "BUY", "pnl": i})
        assert len(memory._working_trades) == 3

    def test_fifo_overflow(self, memory: AgentMemory) -> None:
        """添加 8 条，只保留最新 5 条。"""
        for i in range(8):
            memory.add_trade_result({"action": "BUY", "pnl": i * 10})
        assert len(memory._working_trades) == 5
        # 最老的是第 3 条（index=3），pnl=30
        assert memory._working_trades[0]["pnl"] == 30
        # 最新的是第 7 条，pnl=70
        assert memory._working_trades[-1]["pnl"] == 70


# ── L2 异步操作 ─────────────────────────────────────────

class TestL2AsyncOps:
    """L2 交易记录（通过 FakeRedisBus）。"""

    async def test_save_and_retrieve(self, memory: AgentMemory) -> None:
        """保存后能取回。"""
        await memory.save_trade_to_l2({"action": "BUY", "asset": "BTC-PERP", "pnl": 100})
        trades = await memory.get_recent_trades(count=5)
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"

    async def test_trade_count(self, memory: AgentMemory) -> None:
        """交易数正确计数。"""
        for i in range(3):
            await memory.save_trade_to_l2({"action": "BUY", "pnl": i})
        count = await memory.get_trade_count()
        assert count == 3


# ── L3 异步操作 ─────────────────────────────────────────

class TestL3AsyncOps:
    """L3 反思总结（通过 FakeRedisBus）。"""

    async def test_add_and_retrieve_reflection(self, memory: AgentMemory) -> None:
        await memory.add_reflection("第一次反思总结")
        reflections = await memory._get_reflections(count=5)
        assert len(reflections) == 1
        assert reflections[0] == "第一次反思总结"


# ── get_context_for_decision ────────────────────────────

class TestGetContextForDecision:
    """决策上下文拼接。"""

    async def test_empty_context_format(self, memory: AgentMemory) -> None:
        """无数据时返回结构化空内容。"""
        ctx = await memory.get_context_for_decision()
        assert "WORKING MEMORY" in ctx
        assert "EPISODIC MEMORY" in ctx
        assert "SEMANTIC MEMORY" in ctx
        assert "No data yet" in ctx

    async def test_with_ticks_and_trades(self, memory: AgentMemory) -> None:
        """有 tick 和交易时包含相应数据。"""
        memory.add_tick({"price": 67000})
        memory.add_tick({"price": 67100})
        memory.add_trade_result({"action": "BUY", "asset": "BTC-PERP", "pnl": 50})
        ctx = await memory.get_context_for_decision()
        assert "67000" in ctx
        assert "67100" in ctx
        assert "BUY" in ctx
        assert "BTC-PERP" in ctx

    async def test_context_includes_l2_trades(self, memory: AgentMemory) -> None:
        """L2 历史交易也包含在上下文中。"""
        await memory.save_trade_to_l2({
            "action": "SELL", "asset": "ETH-PERP",
            "entry_price": 3500, "exit_price": 3600, "pnl": 100,
        })
        ctx = await memory.get_context_for_decision()
        assert "ETH-PERP" in ctx
        assert "EPISODIC MEMORY" in ctx

    async def test_context_includes_reflections(self, memory: AgentMemory) -> None:
        """L3 反思也包含在上下文中。"""
        await memory.add_reflection("我倾向于过早止损")
        ctx = await memory.get_context_for_decision()
        assert "过早止损" in ctx
        assert "SEMANTIC MEMORY" in ctx

    async def test_relevant_trades_with_reasoning(self, memory: AgentMemory) -> None:
        """有 reasoning 的交易应通过 TF-IDF 被优先检索。"""
        await memory.save_trade_to_l2({
            "action": "BUY", "asset": "ETH-PERP",
            "reasoning": "ETH showing strong momentum with DeFi growth",
        })
        await memory.save_trade_to_l2({
            "action": "BUY", "asset": "BTC-PERP",
            "reasoning": "BTC bullish pattern RSI oversold",
        })
        trades = await memory.get_relevant_trades("BTC-PERP", "BUY", count=2)
        # BTC 交易应排在 ETH 前面（asset 匹配 + reasoning 匹配）
        assert len(trades) >= 1
        assert trades[0].get("asset") == "BTC-PERP"

    async def test_relevant_trades_empty(self, memory: AgentMemory) -> None:
        """无交易记录时返回空列表。"""
        trades = await memory.get_relevant_trades("BTC-PERP", "BUY", count=5)
        assert trades == []

    async def test_relevant_trades_single_trade(self, memory: AgentMemory) -> None:
        """只有 1 条交易时也能正常返回。"""
        await memory.save_trade_to_l2({"action": "BUY", "asset": "BTC-PERP"})
        trades = await memory.get_relevant_trades("BTC-PERP", "BUY", count=5)
        assert len(trades) == 1
