# agent — Agent 核心

## 用途
交易 Agent 的基类、决策循环、三层记忆、反思模块。

## 文件清单
- `base_agent.py` — Agent 基类，asyncio 生命周期管理（~63行）
- `trading_agent.py` — 核心交易 Agent，LLM 决策循环（~200行）
- `memory.py` — 三层记忆系统：Working / Episodic / Semantic（~126行）
- `reflection.py` — 交易反思模块，每 10 笔触发（~73行）

## 依赖关系
- 本目录依赖：personality/, market/, execution/, integration/
- 被以下模块依赖：main.py
