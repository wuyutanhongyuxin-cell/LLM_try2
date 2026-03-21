# agent — Agent 核心

## 用途
交易 Agent 的基类、决策循环、四层记忆、反思模块。

## 文件清单
- `base_agent.py` — Agent 基类，asyncio 生命周期管理（~63行）
- `trading_agent.py` — 核心交易 Agent，LLM 决策循环+交易计数 Redis 持久化（~264行）
- `memory.py` — 四层记忆系统：Working / Episodic / Semantic / Wisdom（~194行）
- `long_term_memory.py` — L4 永久记忆：归档+投票淘汰+智慧压缩（~171行）
- `memory_pruner.py` — 记忆投票淘汰：10轮LLM投票+≥500笔交易门槛（~139行）
- `multi_sample.py` — 多采样投票决策：多数票+安全 confidence 解析（~52行）
- `reflection.py` — 交易反思模块，每 10 笔触发（~137行）

## 记忆架构
| 层 | 名称 | 存储 | 容量 | 说明 |
|---|------|------|------|------|
| L1 | Working | 内存 | 20 tick + 5 trade | 即时上下文 |
| L2 | Episodic | Redis | 50 笔滚动 | 交易明细 |
| L3 | Semantic | Redis | 20 条滚动 | 反思摘要 |
| L4 | Wisdom | 本地文件 | **无上限** | 永久交易智慧 |

## 交易计数持久化
- 交易计数（`_trade_count`）通过 Redis key `agent:{id}:trade_count` 持久化
- 每次 BUY/SELL 成功后写入，启动时自动恢复（HOLD 不计入）
- 确保 L3 反思（每10笔）和 L4 智慧提取（每500笔）阈值跨重启延续

## 依赖关系
- 本目录依赖：personality/, market/, execution/, integration/
- 被以下模块依赖：main.py, scripts/live_lighter.py
