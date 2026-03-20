# agent — Agent 核心

## 用途
交易 Agent 的基类、决策循环、四层记忆、反思模块。

## 文件清单
- `base_agent.py` — Agent 基类，asyncio 生命周期管理（~63行）
- `trading_agent.py` — 核心交易 Agent，LLM 决策循环（~200行）
- `memory.py` — 四层记忆系统：Working / Episodic / Semantic / Wisdom（~194行）
- `long_term_memory.py` — L4 永久长期记忆：归档+LLM智慧压缩（~147行）
- `reflection.py` — 交易反思模块，每 10 笔触发（~137行）

## 记忆架构
| 层 | 名称 | 存储 | 容量 | 说明 |
|---|------|------|------|------|
| L1 | Working | 内存 | 20 tick + 5 trade | 即时上下文 |
| L2 | Episodic | Redis | 50 笔滚动 | 交易明细 |
| L3 | Semantic | Redis | 20 条滚动 | 反思摘要 |
| L4 | Wisdom | 本地文件 | **无上限** | 永久交易智慧 |

## 依赖关系
- 本目录依赖：personality/, market/, execution/, integration/
- 被以下模块依赖：main.py, scripts/live_lighter.py
