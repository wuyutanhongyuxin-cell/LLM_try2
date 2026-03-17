from __future__ import annotations

"""共用 fixture：FakeRedisBus、sample OceanProfile、global_config 等。"""

import json
from typing import Any

import pytest

from src.personality.ocean_model import OceanProfile


# ── 全局资产配置（加密货币） ────────────────────────────────

GLOBAL_CONFIG: dict[str, list[str]] = {
    "major_assets": ["BTC-PERP", "ETH-PERP"],
    "all_assets": ["BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "DOGE-PERP"],
}

# ── CME 期货资产配置 ──────────────────────────────────────

CME_GLOBAL_CONFIG: dict[str, list[str]] = {
    "major_assets": ["ES", "NQ"],
    "all_assets": ["ES", "NQ", "CL", "GC", "SI", "ZB"],
}


@pytest.fixture()
def global_config() -> dict[str, list[str]]:
    """交易资产全局配置（加密货币）。"""
    return GLOBAL_CONFIG.copy()


@pytest.fixture()
def cme_global_config() -> dict[str, list[str]]:
    """交易资产全局配置（CME 期货）。"""
    return CME_GLOBAL_CONFIG.copy()


# ── 示例 OceanProfile ────────────────────────────────────

@pytest.fixture()
def sample_profile() -> OceanProfile:
    """中庸型人格，用于通用测试。"""
    return OceanProfile(
        name="测试中庸型",
        openness=50, conscientiousness=50,
        extraversion=50, agreeableness=50, neuroticism=50,
    )


# ── FakeRedisBus（替代真实 Redis） ───────────────────────

class FakeRedisBus:
    """内存中的 Redis 替身，实现 AgentMemory 所需的全部接口。"""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}  # key -> list[json_str]
        self._kv: dict[str, str] = {}
        self._published: list[tuple[str, dict]] = []  # (channel, data)

    # ── pub/sub ──

    async def publish(self, channel: str, data: dict[str, Any]) -> None:
        self._published.append((channel, data))

    # ── list 操作 ──

    async def lpush_json(self, key: str, value: Any) -> None:
        if key not in self._store:
            self._store[key] = []
        self._store[key].insert(0, json.dumps(value, default=str))

    async def lrange_json(self, key: str, start: int, end: int) -> list[Any]:
        lst = self._store.get(key, [])
        # Redis lrange end 是闭区间
        sliced = lst[start: end + 1]
        return [json.loads(item) for item in sliced]

    async def ltrim(self, key: str, start: int, end: int) -> None:
        if key in self._store:
            self._store[key] = self._store[key][start: end + 1]

    # ── kv（AgentMemory.get_trade_count 需要 _ensure_connected） ──

    async def _ensure_connected(self) -> "FakeRedisClient":
        """返回一个假 Redis 客户端，支持 llen。"""
        return FakeRedisClient(self._store)

    # ── set/get ──

    async def set_json(self, key: str, value: Any) -> None:
        self._kv[key] = json.dumps(value, default=str)

    async def get_json(self, key: str) -> Any | None:
        raw = self._kv.get(key)
        return json.loads(raw) if raw is not None else None


class FakeRedisClient:
    """FakeRedisBus._ensure_connected 返回的假 Redis 客户端。"""

    def __init__(self, store: dict[str, list[str]]) -> None:
        self._store = store

    async def llen(self, key: str) -> int:
        return len(self._store.get(key, []))


@pytest.fixture()
def fake_redis() -> FakeRedisBus:
    """返回一个全新的 FakeRedisBus 实例。"""
    return FakeRedisBus()
