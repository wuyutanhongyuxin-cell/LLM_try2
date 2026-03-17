# 基于人格的多 Agent 加密货币交易系统

> **[English](README.md) | 中文**

> 用心理学 Big Five (OCEAN) 人格模型驱动的多 Agent 加密货币纸上交易系统——每个 Agent 拥有独特性格，性格决定交易风格

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-193%20passed-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 项目简介

本系统使用 **Big Five 人格理论（OCEAN 模型）** 创建多样化的加密货币交易 Agent。每个 Agent 拥有独特的人格参数组合，这些参数**确定性地**塑造其交易行为——风险承受能力、仓位大小、资产选择和决策频率。

多个 Agent 并行运行，各自通过 LLM（`litellm` 统一接口）独立决策，同时由代码强制执行的硬约束确保任何 Agent 都不会超出其性格所决定的限制。

**核心理念**：LLM 只负责「建议」，代码负责「执行」。性格参数 → 硬编码公式 → 不可逾越的交易约束。

```
                    ┌─────────────────────────────────┐
                    │       OCEAN 人格参数              │
                    │  O=90 C=80 E=25 A=20 N=10       │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌────────────┐ ┌──────────────┐
      │ System Prompt│ │  硬约束     │ │ 三层记忆      │
      │ （性格注入）  │ │ （代码强制）│ │ W / E / S    │
      └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
             │               │               │
             └───────┬───────┘               │
                     ▼                       │
              ┌──────────────┐               │
              │  LLM 调用    │◄──────────────┘
              │ （3次投票）   │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  校验 & Clip │  ← 超限字段被裁剪到合法范围
              │  信号约束     │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  纸上交易     │  → PnL - 成本（滑点 + 手续费 + 资金费率）
              └──────────────┘
```

---

## 核心设计原则

| 原则 | 实现方式 |
|------|---------|
| **LLM 建议，代码执行** | `_validate_signal()` 将所有值裁剪到约束范围内 |
| **人格 = 连续维度，非类型标签** | OCEAN 分数 0-100 连续值，不是 MBTI 类型 |
| **确定性约束** | `trait_to_constraint.py` 使用固定公式，不受 LLM 影响 |
| **Agent 完全隔离** | 每个 Agent 独立记忆、独立持仓、独立 PnL |
| **金额精确计算** | 所有金额使用 `decimal.Decimal`，禁止 float 算钱 |
| **真实成本模拟** | 每笔交易扣除滑点 + Taker 手续费 + 8h 资金费率 |
| **多采样一致性** | 每次决策调用 3 次 LLM，多数投票决定方向 |
| **防回溯偏差** | 资产匿名化阻止 LLM 回忆历史价格走势 |

---

## OCEAN 人格模型与交易映射

五个维度如何影响交易行为：

| 维度 | 缩写 | 高分 (→100) | 低分 (→0) |
|------|------|-------------|-----------|
| **开放性** | O | 探索山寨币、新策略、高波动资产 | 只做 BTC/ETH，保守策略 |
| **尽责性** | C | 严格止损、纪律执行、规则至上 | 冲动交易，忽视风控 |
| **外向性** | E | 追涨杀跌、跟随市场情绪 | 逆向交易、独立判断 |
| **宜人性** | A | 从众跟风、与市场共识一致 | 对抗市场共识、倾向做空 |
| **神经质** | N | 极度厌恶损失、止损极紧、频繁割肉 | 情绪稳定、能 hold 住回撤 |

### 约束映射公式（硬编码，LLM 不可覆盖）

```python
# 神经质(N)越高 → 仓位越小、止损越紧、检查越频繁
max_position_pct     = clip(5 + (100 - N) * 0.25, 5, 30)
stop_loss_pct        = clip(1 + (100 - N) * 0.14, 1, 15)
max_drawdown_pct     = clip(2 + (100 - N) * 0.18, 2, 20)
rebalance_interval   = N>70 → 5分钟, N>40 → 1小时, 否则 → 1天

# 开放性(O)越高 → 交易更多币种、同时持仓更多
max_concurrent_pos   = clip(1 + O // 20, 1, 6)
allowed_assets       = O>60 → 全部币种, 否则 → 仅主流(BTC/ETH)

# 外向性(E) → 情绪数据 + 动量权重
use_sentiment        = E > 50
momentum_weight      = E / 100

# 宜人性(A)越低 → 越倾向逆向交易
contrarian_weight    = (100 - A) / 100

# 尽责性(C)越高 → 越严格的止损和信心要求
require_stop_loss    = C > 50
min_confidence       = clip(C * 0.008, 0.2, 0.8)
```

---

## 7 个预定义人格原型

| 原型 | O | C | E | A | N | 交易风格 |
|------|---|---|---|---|---|---------|
| **冷静创新型** | 90 | 80 | 25 | 20 | 10 | 探索新资产、纪律严明、逆向思维 |
| **保守焦虑型** | 15 | 85 | 20 | 70 | 90 | 只做 BTC/ETH、极紧止损、每 5 分钟检查 |
| **激进冒险型** | 85 | 20 | 80 | 15 | 10 | 全币种、追涨杀跌、风控宽松 |
| **纪律动量型** | 50 | 90 | 75 | 50 | 30 | 跟随趋势但严格纪律 |
| **逆向价值型** | 60 | 75 | 10 | 10 | 25 | 逆势而行、耐心等待、价值导向 |
| **平衡中庸型** | 50 | 50 | 50 | 50 | 50 | 各维度均衡，中规中矩 |
| **情绪追涨型** | 70 | 15 | 90 | 80 | 75 | FOMO 驱动、从众跟风、紧止损 |

---

## 项目结构

```
personality-trading-agents/
├── config/
│   ├── agents.yaml              # Agent 人格配置（OCEAN 参数）
│   ├── trading.yaml             # 交易参数 + 成本配置 + 风控 + 匿名化 + 辩论开关
│   ├── llm.yaml                 # LLM 配置 + 多采样 + 限流
│   └── market_knowledge.json    # 市场因果关系知识图谱
├── src/
│   ├── personality/             # 人格引擎：OCEAN 模型 + 约束映射 + Prompt 生成（含版本hash）
│   ├── agent/                   # Agent 核心：交易 Agent + 多采样投票 + 三层记忆 + 反思
│   ├── market/                  # 行情数据：Mock/Live 数据源 + 技术指标 + 对抗性场景
│   ├── execution/               # 执行层：信号 + 纸上交易 + 聚合 + 风控 + 成本 + 漂移 + 辩论 + 策略
│   ├── integration/             # 外部集成：Redis 消息总线 + Telegram（信号+漂移+成本告警）
│   ├── utils/                   # 工具：配置 + 日志 + 匿名化 + 全链路日志 + TF-IDF + 知识图谱
│   └── main.py                  # 主入口
├── tests/                       # 193 个测试，覆盖全部模块
├── scripts/
│   ├── dashboard.py             # Rich 终端实时仪表盘
│   ├── backtest.py              # 规则回测
│   ├── llm_backtest.py          # LLM 真实回测（多次运行 + 一致性 + 多市况）
│   ├── generate_synthetic_data.py # 合成多市况数据（熊市/横盘/牛市）
│   ├── export_training_data.py  # 决策轨迹导出（JSONL 微调格式）
│   └── create_agents_config.py  # 批量生成 Agent 配置
└── pyproject.toml
```

---

## 系统加固（Phase A-F）

基于学术论文的系统性审查（TradeTrap、Profit Mirage、FINSABER、tau-bench、LiveTradeBench）后进行的全面加固：

### A. 真实回测引擎

**交易成本模型**（`cost_model.py`）：每笔交易扣除真实世界成本：

| 成本项 | 默认值 | 来源 |
|--------|--------|------|
| 滑点 | 5 bps (0.05%) | 市场微观结构 |
| Taker 手续费 | 0.04% | Binance 永续合约 |
| Maker 手续费 | 0.02% | Binance 永续合约 |
| 资金费率 | 0.015% / 8h | 保守估计，2024 实际约 0.01-0.017%/8h，BitMEX 78% 时间锚定 0.01% |

**资产匿名化**（`anonymizer.py`）：Prompt 中将 `BTC-PERP` 替换为 `ASSET_A`，防止 LLM 回忆历史价格。Profit Mirage (2025) 实测去除名称偏差后 Sharpe 衰减 51-62%。

**LLM 真实回测**（`llm_backtest.py`）：
```bash
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 3 --agents 3 --anonymize
```
输出：各 Agent 的平均 PnL、PnL 标准差、action 一致率、pass^k 指标。

### B. Agent 决策稳定性

**多采样投票**（`multi_sample.py`）：每次决策调用 LLM 3 次，多数票决定方向。基于 Self-Consistency（Wang et al., ICLR 2023）：1→3 次采样捕获约 80% 一致性增益。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decision_samples` | 3 | 每次决策的 LLM 调用次数 |
| `consensus_threshold` | 0.6 | 多数票占比阈值，低于此值默认 HOLD |

**行为漂移检测**（`consistency_monitor.py`）：用 KL 散度监控 action 分布变化，三级告警：

| 严重程度 | KL 阈值 | 动作 |
|---------|---------|------|
| 警告 | > 0.1 | 记录日志 |
| 严重 | > 0.2 | Telegram 告警 |
| 暂停 | > 0.5 | 暂停该 Agent 交易 |

**Prompt 版本追溯**：每个 System Prompt 末尾附加 SHA-256 hash（`[prompt_version: abc123...]`），存入 `TradeSignal.prompt_hash`，支持完整回溯。

### C. 记忆系统升级

**TF-IDF 混合检索**（替代纯规则评分）：L2 情节记忆使用 TF-IDF 语义相似度(50%) + 规则评分(50%)——同资产(+0.5)、同 action(+0.33)、有盈亏(+0.17)——加上时间衰减(0.95^position)。纯 Python 实现，无需 sklearn/numpy。

**指数衰减**：L3 语义记忆应用衰减权重（alpha=0.98）。近期反思完整展示，远期反思只显示前 50 字符。

### D. 数据层修复

**精确 24h 变化**：MockDataFeed 使用 24 条前价格计算 24h 变化（1h K 线 × 24），替代原来单根 K 线 open→close 的严重失真。

**MarketSnapshot 扩展**：新增 `open_price` 和 `funding_rate` 字段。

### E. Agent 级风控

全局风控新增 `check_agent_risk()`：监控单个 Agent 的回撤和连续亏损，可暂停单个 Agent 而不影响全局。

### F. 可观测性

**全链路交易日志**（`trade_logger.py`）：每笔交易记录完整决策链——行情快照、Prompt hash、LLM 原始响应（前 500 字符）、校验前后信号对比、被 clip 字段列表、执行结果、成本明细。

**新增 Telegram 告警**：行为漂移告警、成本报告。

---

## P0：对抗性压力测试（TradeTrap 启发）

**对抗性场景生成器**（`adversarial.py`）：5 种基于真实 BTC 极端事件的压力测试场景：

| 场景 | 真实原型 | 效果 |
|------|---------|------|
| 闪崩 | 2024.3.19 BitMEX：$67K→$8.9K（2分钟，仅现货） | 单根 K 线 -15% |
| 暴涨 | 2024.12.5 BTC 突破 $100K | 连续 3 根 +5% |
| 假突破 | 2024 Q1 Grayscale GBTC 抛售期 | 先涨+5% 再跌-9%，净-4% |
| 极端横盘 | 2023 Q3 BTC $25K-$30K 区间 50 天 | 每根 ±1% 随机 |
| V 型反转 | 2024.12 BTC $100K→$93K→$100K | 先跌-6% 再涨+6.5% |

**多市况回测**：从单个 CSV 生成熊市/横盘/牛市合成数据，跨市况对比：
```bash
python scripts/generate_synthetic_data.py --csv data/btc_1h_2024.csv --output data/
python scripts/llm_backtest.py --csv data/btc_bull.csv --runs 3 --multi-market --anonymize
```

---

## P1：高级记忆 + 元反思

**两层反思机制**：在常规反思（每 10 笔交易）之上，每 30 笔交易触发**元反思**——分析多次反思之间的模式、策略演化和反复出现的盲点。元反思以 `[META]` 标记存入 L3 记忆。

**TF-IDF 记忆检索**（`tfidf.py`）：纯 Python 实现，替代手写规则。结合语义和规则评分：
- TF-IDF 余弦相似度计算交易 reasoning 文本匹配（50% 权重）
- 规则加分：同资产(+0.5)、同 action(+0.33)、有盈亏(+0.17)（50% 权重）
- 时间衰减：0.95^position（越新的交易权重越高）

---

## P2：市场知识图谱 + 微调数据导出

**轻量知识图谱**（`market_knowledge.json`）：纯 JSON 实现的市场因果关系图谱，覆盖宏观、链上、衍生品因子，不需要 Neo4j 或任何图数据库。

| 因子 | 对 BTC 影响 | 强度 | 滞后 |
|------|------------|------|------|
| 美联储利率 | 负相关 | 强 | ~30天 |
| 全球 M2 供应量 | 正相关 | 强 | ~90天 |
| 美元指数 DXY | 负相关 | 中等 | ~7天 |
| 恐慌指数 VIX | 负相关 | 中等 | 0天 |
| BTC ETF 资金流 | 正相关 | 强 | ~1天 |
| 交易所储备量 | 负相关（下降=看涨） | 中等 | ~3天 |
| 资金费率 | 反向指标 | 弱 | 0天 |
| 未平仓合约 | 放大波动 | 中等 | 0天 |

来源：Fidelity Digital Assets 研报（BTC-M2 相关系数 r=0.78，~90天滞后）、S&P Global 研报、Frontiers in Blockchain 2025。

知识上下文在每次 Decision Prompt 中注入（位于记忆段之前），为 Agent 提供宏观层面的市场认知。

**微调数据导出**（`export_training_data.py`）：将成功交易决策导出为 JSONL 格式训练数据，用于 LLM 微调（LoRA/QLoRA）。使用校验后信号（validated behavior）作为训练目标。

```bash
python scripts/export_training_data.py --agent agent_calm_innovator --output data/finetune/
```

---

## P3：Bull/Bear 辩论 + 执行策略抽象

**Bull/Bear 辩论**（`debate.py`）：受 TradingAgents (arxiv 2412.20138) 启发，当 voting 模式开启 `enable_debate: true` 时：

1. 收集所有 Agent 的 reasoning，分为 Bull(BUY) / Bear(SELL) / Neutral(HOLD) 三组
2. 裁判 LLM 输出：`dominant_view`、`confidence_adjustment`(±0.3)、`key_argument`、`risk_flag`
3. 如果 BULL 主导：BUY 信号信心提升，SELL 信号信心降低；反之亦然
4. 只调整信心权重，不改变交易方向

**重要**：辩论不共享 Agent 记忆，只使用信号中的公开 reasoning 字段。

**执行策略抽象**（`strategy.py`）：将信号校验逻辑从 Agent 核心解耦：

```
ExecutionStrategy (抽象基类)
  └── RuleBasedStrategy    ← 当前默认（OCEAN 约束 clip 逻辑）
  └── RLStrategy           ← 未来 Phase 2（LLM + RL 混合架构）
```

`_build_signal_from_data()` 现在委托给 `strategy.process_signal()`，未来可直接替换为 RL 策略，无需修改 Agent 代码。

---

## 三层记忆系统（FinMem 启发）

| 层级 | 名称 | 内容 | 容量 | 存储 | 检索方式 |
|------|------|------|------|------|---------|
| L1 | 工作记忆 | 最近 20 条 tick + 最近 5 次交易结果 | 20+5 | 内存 | 全量（每次决策） |
| L2 | 情节记忆 | 完整交易记录（价格、盈亏、reasoning） | 50 笔 | Redis | TF-IDF 混合检索 |
| L3 | 语义记忆 | 反思总结（自然语言） | 20 条 | Redis | 指数衰减加权 |

**记忆隔离**：每个 Agent 的记忆完全独立，互不共享，确保性格差异不被稀释。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API 密钥：
# ANTHROPIC_API_KEY=sk-ant-...
# REDIS_URL=redis://localhost:6379/0
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
```

### 3. 运行测试

```bash
pytest tests/ -v
# 应该看到 193 passed
```

### 4. 启动系统

```bash
python -m src.main
```

### 5. 运行 LLM 回测（推荐开启匿名化）

```bash
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 3 --anonymize
```

### 6. 启动仪表盘（另开一个终端）

```bash
python scripts/dashboard.py
```

---

## 配置说明

### 交易成本（`config/trading.yaml`）
```yaml
costs:
  slippage_bps: 5              # 滑点 5 bps = 0.05%
  taker_fee_rate: 0.0004       # Taker 手续费 0.04%
  maker_fee_rate: 0.0002       # Maker 手续费 0.02%
  funding_rate_8h: 0.00015     # 资金费率 0.015%/8h
  enable_costs: true           # false 可关闭（对比实验用）
anonymize: false               # 回测建议开启 true
aggregator:
  enable_debate: false         # true 启用 Bull/Bear 辩论（仅 voting 模式）
```

### 多采样投票（`config/llm.yaml`）
```yaml
decision_samples: 3            # 每次决策 LLM 调用次数
consensus_threshold: 0.6       # 投票占比阈值
max_calls_per_minute: 20       # 全局限流
max_cost_per_backtest_usd: 50  # 回测成本硬上限
```

---

## 信号聚合模式

| 模式 | 说明 | 使用场景 |
|------|------|---------|
| `independent` | 每个 Agent 信号独立执行，各自计算 PnL | 对比实验：哪种性格表现最好 |
| `voting` | 按 `信心度 x 历史Sharpe` 加权投票 | 集成决策：综合多个性格的智慧 |
| `voting` + 辩论 | Bull/Bear 辩论调整信心权重后再投票 | 平衡型集成决策 |

**Bull/Bear 辩论**（`debate.py`）：启用 `enable_debate: true` 后，裁判 LLM 评估所有 Agent 的 reasoning，分为 Bull(BUY)/Bear(SELL)/Neutral(HOLD) 三组，输出 `confidence_adjustment`（±0.3）调整信号权重，但不改变交易方向。灵感来自 TradingAgents (arxiv 2412.20138)。

**重要**：辩论不共享 Agent 记忆，只使用信号中的公开 `reasoning` 字段。

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.9+ | asyncio 异步生态 |
| LLM 接口 | `litellm` | 统一接口，支持 Claude/GPT/本地模型 |
| 数据校验 | Pydantic v2 | 类型安全 + 序列化 |
| 消息队列 | Redis pub/sub | Agent 信号广播 |
| 通知推送 | aiogram 3.x | Telegram 实时告警 + 漂移告警 |
| 日志 | loguru | 结构化彩色输出 |
| 仪表盘 | rich | 终端实时 UI |
| 测试 | pytest + pytest-asyncio | 193 个测试，全模块覆盖 |

**刻意不用的依赖**：pandas、numpy、django、flask、sqlalchemy——保持轻量。

---

## Agent 决策流程详解

每个 Agent 是独立的 `asyncio.Task`，主循环如下：

```
循环:
  1. 等待 rebalance_interval 秒（由 N 维度决定）
  2. 获取最新行情 → MarketSnapshot
  3. 从三层记忆提取决策上下文（相关性检索 + 衰减加权）
  4. [如果启用匿名化] 将 BTC-PERP → ASSET_A
  5. 生成 Decision Prompt（行情 + 持仓 + 记忆 + 总资产）
  6. 调用 LLM × 3 次 → 多数投票（60% 阈值）
  7. [如果启用匿名化] 将 ASSET_A → BTC-PERP
  8. 校验 & Clip（关键步骤）：
     - action 必须是 BUY/SELL/HOLD
     - asset 必须在允许列表中（否则拒绝）
     - size_pct 裁剪到 [0, max_position_pct]
     - confidence 裁剪到 [0, 1]
     - 如果 require_stop_loss=True 但未设止损 → 拒绝
     - 记录 prompt_hash + llm_model 到信号中
  9. 如果 confidence >= min_confidence_threshold：
     - 扣除交易成本（滑点 + 手续费）
     - 发布信号到 Redis
     - 记录全链路日志
     - 更新 L1 + L2 记忆
     - 检查行为漂移
 10. 每 10 笔交易 → 触发反思 → 更新 L3 记忆
 11. 每 30 笔交易 → 触发元反思（分析多次反思的模式）→ [META] 标记存入 L3
```

---

## Telegram 通知

系统推送以下事件：
- 交易信号（含完整决策理由）
- 止损/止盈触发
- Agent 反思报告（每 10 笔交易）
- 每日排行榜汇总
- **行为漂移告警**（KL 散度超阈值时）
- **成本报告**（每个 Agent 累计交易成本）

通知示例：
```
🧠 冷静创新型 (O90/C80/E25/A20/N10)
📊 BUY BTC-PERP @ $67,200
💰 Size: 25% | SL: $64,000 | TP: $72,000
🎯 Confidence: 0.85
💭 链上数据显示鲸鱼积累，RSI 超卖
🔑 主导维度: O—愿意在回调中建仓新头寸

⚠️ 行为漂移告警 [CRITICAL]
Agent: 激进冒险型
Action KL=0.312 > critical(0.2)

📈 Daily Report - 2026-03-17
| # | Agent      | PnL    | Sharpe | MaxDD  | Trades | 成本  |
|---|-----------|--------|--------|--------|--------|-------|
| 1 | 冷静创新型 | +$320  | 1.85   | -3.2%  | 5      | $12.3 |
| 2 | 逆向价值型 | +$180  | 1.42   | -2.1%  | 3      | $7.8  |
| 3 | 激进冒险型 | -$450  | -0.32  | -12.5% | 12     | $28.5 |
```

---

## 开发路线图

### Phase 1（完成）：纸上交易验证
- [x] OCEAN 人格模型 + 7 个原型
- [x] 确定性约束映射
- [x] LLM 驱动决策 + 硬约束强制执行
- [x] 三层记忆系统（相关性检索 + 衰减）
- [x] 纸上交易 + 完整绩效跟踪
- [x] 信号聚合（独立 + 投票模式）
- [x] 全局 + Agent 级风控
- [x] Telegram 通知 + 漂移告警
- [x] Rich 终端仪表盘
- [x] 历史回测（规则 + LLM 驱动）

### Phase 1.5（完成）：系统加固
- [x] 交易成本模型（滑点 + 手续费 + 资金费率）
- [x] 多采样投票（3 次 LLM，60% 共识阈值）
- [x] 资产匿名化（防回溯偏差）
- [x] 行为漂移检测（三级 KL 阈值）
- [x] Prompt 版本追溯（SHA-256）
- [x] 全链路交易日志
- [x] 相关性记忆检索
- [x] 指数记忆衰减

### P0（完成）：对抗性测试 + 多市况回测
- [x] 5 种对抗性场景（闪崩/暴涨/假突破/横盘/V 反转），基于真实 BTC 事件
- [x] MockDataFeed 对抗性场景注入支持
- [x] 合成数据生成（从单 CSV 生成熊市/横盘/牛市）
- [x] `--multi-market` 模式 + 跨市况对比表
- [x] 回测成本硬上限执行（`max_cost_per_backtest_usd`）

### P1（完成）：高级记忆 + 元反思
- [x] 两层反思：L1 反思（每 10 笔）+ L2 元反思（每 30 笔）
- [x] 元反思分析反思间的模式，识别策略演化和盲点
- [x] 纯 Python TF-IDF 引擎（无 sklearn/numpy）
- [x] 混合检索：TF-IDF 语义相似度(50%) + 规则评分(50%) + 时间衰减

### P2（完成）：知识图谱 + 微调数据导出
- [x] 轻量市场知识图谱（`market_knowledge.json`）— BTC/ETH 因果关系（宏观/链上/衍生品）
- [x] 知识上下文注入 Decision Prompt（位于记忆段之前）
- [x] 决策轨迹导出脚本（`export_training_data.py`）— JSONL 格式，支持 OpenAI/Qwen 微调

### P3（完成）：Bull/Bear 辩论 + 执行策略抽象
- [x] Bull/Bear 辩论模块（`debate.py`）— 裁判 LLM 评估 reasoning，调整信心权重
- [x] `enable_debate` 开关（`config/trading.yaml`，默认关闭）
- [x] ExecutionStrategy 接口 + RuleBasedStrategy（`strategy.py`）— 校验逻辑从 Agent 解耦
- [x] 未来 RL 策略可直接替换 RuleBasedStrategy，无需修改 Agent 代码

### Phase 2（未来）：实盘交易
- [ ] 接入真实 DEX（GRVT/Paradex）
- [ ] 人格动态进化（反思驱动自动调参）
- [ ] 情绪数据源（Twitter/Telegram sentiment）
- [ ] 投票模式实盘验证
- [ ] RL 策略替换 RuleBasedStrategy

---

## 学术参考

本系统加固受以下论文启发：
- **Profit Mirage** (2025)：LLM 交易 Agent 因回溯偏差 Sharpe 衰减 51-62%
- **Self-Consistency** (Wang et al., ICLR 2023)：多采样投票在 3 次采样时捕获约 80% 一致性增益
- **TradeTrap** (2025)：不计成本的回测收益虚高 2-5 倍
- **tau-bench** (2025)：pass@1=61% 但 pass^8=25%——单次运行结果不可靠
- **FinMem** (2023)：三层记忆 + 相关性评分 + 衰减机制
- **TradingAgents** (2024, arxiv 2412.20138)：Bull/Bear 研究员辩论机制
- **TradingGroup** (2025, arxiv 2508.17565)：决策轨迹收集用于 LLM 微调
- **Fidelity Digital Assets** (2024)：BTC-M2 相关系数 r=0.78，约 90 天滞后

---

## 许可证

MIT
