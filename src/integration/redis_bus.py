from __future__ import annotations

"""Redis pub/sub 消息总线封装。

提供发布/订阅 + KV/List 操作，供 Agent 信号广播和 Memory 持久化使用。
"""

import json
import os
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from loguru import logger


class RedisBus:
    """Redis 发布/订阅 + KV 封装，支持自动重连。"""

    def __init__(self, url: str | None = None) -> None:
        """初始化连接参数。url 默认从环境变量 REDIS_URL 读取。"""
        self._url: str = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """建立 Redis 连接。"""
        try:
            self._redis = aioredis.from_url(self._url, decode_responses=True)
            await self._redis.ping()
            logger.info("Redis 已连接: {}", self._url)
        except Exception as exc:
            logger.error("Redis 连接失败: {}", exc)
            self._redis = None
            raise

    async def disconnect(self) -> None:
        """关闭 Redis 连接。"""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis 已断开")

    async def _ensure_connected(self) -> aioredis.Redis:
        """确保连接可用，断开时自动重连。"""
        if self._redis is None:
            await self.connect()
        assert self._redis is not None  # noqa: S101
        return self._redis

    async def publish(self, channel: str, data: dict[str, Any]) -> None:
        """发布消息到指定频道（dict -> JSON 序列化）。"""
        try:
            r = await self._ensure_connected()
            await r.publish(channel, json.dumps(data, default=str))
        except Exception as exc:
            logger.error("Redis publish 失败 [{}]: {}", channel, exc)
            self._redis = None

    async def subscribe(self, channel: str) -> AsyncIterator[dict[str, Any]]:
        """订阅频道，返回异步迭代器（JSON -> dict 反序列化）。"""
        r = await self._ensure_connected()
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        logger.info("已订阅频道: {}", channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    yield json.loads(message["data"])
                except json.JSONDecodeError:
                    logger.warning("JSON 解析失败，跳过: {}", message["data"])
        finally:
            await pubsub.unsubscribe(channel)

    async def set_json(self, key: str, value: Any) -> None:
        """写入 JSON 值。"""
        try:
            r = await self._ensure_connected()
            await r.set(key, json.dumps(value, default=str))
        except Exception as exc:
            logger.error("Redis SET 失败 [{}]: {}", key, exc)
            self._redis = None

    async def get_json(self, key: str) -> Any | None:
        """读取 JSON 值，key 不存在返回 None。"""
        try:
            r = await self._ensure_connected()
            raw = await r.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.error("Redis GET 失败 [{}]: {}", key, exc)
            self._redis = None
            return None

    async def lpush_json(self, key: str, value: Any) -> None:
        """向列表头部插入 JSON 值。"""
        try:
            r = await self._ensure_connected()
            await r.lpush(key, json.dumps(value, default=str))
        except Exception as exc:
            logger.error("Redis LPUSH 失败 [{}]: {}", key, exc)
            self._redis = None

    async def lrange_json(self, key: str, start: int, end: int) -> list[Any]:
        """读取列表指定范围，返回反序列化后的列表。"""
        try:
            r = await self._ensure_connected()
            raw_list = await r.lrange(key, start, end)
            return [json.loads(item) for item in raw_list]
        except Exception as exc:
            logger.error("Redis LRANGE 失败 [{}]: {}", key, exc)
            self._redis = None
            return []

    async def ltrim(self, key: str, start: int, end: int) -> None:
        """裁剪列表，只保留 [start, end] 范围内的元素。"""
        try:
            r = await self._ensure_connected()
            await r.ltrim(key, start, end)
        except Exception as exc:
            logger.error("Redis LTRIM 失败 [{}]: {}", key, exc)
            self._redis = None
