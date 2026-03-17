from __future__ import annotations

"""Agent 基类：生命周期管理。所有 Agent 继承此类，只需实现 _run_loop()。"""

import asyncio
from abc import ABC, abstractmethod

from loguru import logger


class BaseAgent(ABC):
    """所有 Agent 的基类，管理异步生命周期。"""

    def __init__(self, agent_id: str, name: str) -> None:
        """初始化 Agent。

        Args:
            agent_id: 唯一标识，如 "agent_calm_innovator"
            name: 人格原型名称，如 "冷静创新型"
        """
        self._agent_id: str = agent_id
        self._name: str = name
        self._running: bool = False
        self._task: asyncio.Task | None = None

    @property
    def agent_id(self) -> str:
        """Agent 唯一标识。"""
        return self._agent_id

    @property
    def name(self) -> str:
        """人格原型名称。"""
        return self._name

    @property
    def is_running(self) -> bool:
        """Agent 是否正在运行。"""
        return self._running

    async def start(self) -> None:
        """启动 Agent 主循环，创建异步任务。"""
        if self._running:
            logger.warning(f"Agent {self._name} 已在运行，跳过重复启动")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Agent {self._name} ({self._agent_id}) 已启动")

    async def stop(self) -> None:
        """优雅关闭：取消任务并等待退出。"""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Agent {self._name} ({self._agent_id}) 已停止")

    @abstractmethod
    async def _run_loop(self) -> None:
        """主循环，子类必须实现。"""
        ...
