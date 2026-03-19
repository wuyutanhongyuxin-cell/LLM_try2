# 基于人格的多 Agent 加密货币交易系统

> **[English](README.md) | 中文**

> 用心理学 Big Five (OCEAN) 人格模型驱动的多 Agent 加密货币纸上交易系统——每个 Agent 拥有独特性格，性格决定交易风格

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-253%20passed-brightgreen.svg)](#)
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
| **真实成本模拟** | 加密货币：滑点 + 手续费 + 资金费率；CME：滑点 + 按手佣金 |
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

## 32 个预定义人格原型（2^5 二元 OCEAN）

基于 **SLOAN 人格分类体系** —— 将每个 OCEAN 维度二元化为 High/Low，穷举全部 32 种独特交易人格。4 个经典原型保留原始参数，28 个新增原型使用 H=80, L=20。

### 4 个经典原型（★）

| 原型 | O | C | E | A | N | 编码 | 交易风格 |
|------|---|---|---|---|---|------|---------|
| **冷静创新型** ★ | 90 | 80 | 25 | 20 | 10 | HHLLL | 探索新资产、纪律严明、逆向思维 |
| **保守焦虑型** ★ | 15 | 85 | 20 | 70 | 90 | LHLHH | 只做主流品种、极紧止损、5 分钟检查 |
| **激进冒险型** ★ | 85 | 20 | 80 | 15 | 10 | HLHLL | 全品种、追涨杀跌、风控宽松 |
| **情绪追涨型** ★ | 70 | 15 | 90 | 80 | 75 | HLHHH | FOMO 驱动、从众跟风、紧止损 |

### O↓C↓ 保守散漫系（8 型）

| 原型 | O | C | E | A | N | 编码 | 交易风格 |
|------|---|---|---|---|---|------|---------|
| 散漫逆风型 | 20 | 20 | 20 | 20 | 20 | LLLLL | 无纪律、无方向、完全被动 |
| 焦虑叛逆型 | 20 | 20 | 20 | 20 | 80 | LLLLH | 恐惧驱动的逆向、无风控 |
| 随性观望型 | 20 | 20 | 20 | 80 | 20 | LLLHL | 随和但被动、等待共识 |
| 优柔寡断型 | 20 | 20 | 20 | 80 | 80 | LLLHH | 害怕亏损又害怕错过、犹豫不决 |
| 赌徒冲锋型 | 20 | 20 | 80 | 20 | 20 | LLHLL | 鲁莽追涨、无止损、心态稳定 |
| 神经短线型 | 20 | 20 | 80 | 20 | 80 | LLHLH | 追涨后恐慌割肉 |
| 跟风散户型 | 20 | 20 | 80 | 80 | 20 | LLHHL | 乐观从众、无纪律 |
| 恐慌跟风型 | 20 | 20 | 80 | 80 | 80 | LLHHH | 跟风买入后恐慌抛售 |

### O↓C↑ 保守纪律系（7 型 + 1 经典）

| 原型 | O | C | E | A | N | 编码 | 交易风格 |
|------|---|---|---|---|---|------|---------|
| 铁壁防守型 | 20 | 80 | 20 | 20 | 20 | LHLLL | 防守堡垒、严格规则、逆向思维 |
| 谨慎狙击型 | 20 | 80 | 20 | 20 | 80 | LHLLH | 耐心等待、紧止损、极少交易 |
| 稳健保守型 | 20 | 80 | 20 | 80 | 20 | LHLHL | 安全稳健、跟随共识 |
| 纪律突击型 | 20 | 80 | 80 | 20 | 20 | LHHLL | 跟随动量但严格风控 |
| 精算套利型 | 20 | 80 | 80 | 20 | 80 | LHHLH | 精准入场、紧风控、焦虑出场 |
| 纪律跟随型 | 20 | 80 | 80 | 80 | 20 | LHHHL | 有纪律地跟随趋势和共识 |
| 风控趋势型 | 20 | 80 | 80 | 80 | 80 | LHHHH | 全面风控、紧跟共识趋势 |

### O↑C↓ 探索冲动系（6 型 + 2 经典）

| 原型 | O | C | E | A | N | 编码 | 交易风格 |
|------|---|---|---|---|---|------|---------|
| 狂野猎手型 | 80 | 20 | 20 | 20 | 20 | HLLLL | 探索冷门资产、无规则、心态稳 |
| 偏执创新型 | 80 | 20 | 20 | 20 | 80 | HLLLH | 尝试新事物但回撤时恐慌 |
| 佛系探索型 | 80 | 20 | 20 | 80 | 20 | HLLHL | 好奇但被动、随和、佛系 |
| 敏感探路型 | 80 | 20 | 20 | 80 | 80 | HLLHH | 谨慎探索、焦虑驱动出场 |
| 躁动投机型 | 80 | 20 | 80 | 20 | 80 | HLHLH | 高频投机伴随恐慌出场 |
| 乐观冲浪型 | 80 | 20 | 80 | 80 | 20 | HLHHL | 乐观追涨冷门资产、无止损 |

### O↑C↑ 探索纪律系（7 型 + 1 经典）

| 原型 | O | C | E | A | N | 编码 | 交易风格 |
|------|---|---|---|---|---|------|---------|
| 精密逆向型 | 80 | 80 | 20 | 20 | 80 | HHLLH | 纪律逆向、多品种、紧止损 |
| 沉稳研究型 | 80 | 80 | 20 | 80 | 20 | HHLHL | 深度分析、耐心持有、共识感知 |
| 审慎观察型 | 80 | 80 | 20 | 80 | 80 | HHLHH | 谨慎、有纪律、焦虑调节 |
| 全能主导型 | 80 | 80 | 80 | 20 | 20 | HHHLL | 全方位：探索、纪律、主导 |
| 高压精英型 | 80 | 80 | 80 | 20 | 80 | HHHLH | 高压下高表现、紧止损 |
| 完美趋势型 | 80 | 80 | 80 | 80 | 20 | HHHHL | 理想趋势跟随者：纪律+冷静 |
| 全面紧绷型 | 80 | 80 | 80 | 80 | 80 | HHHHH | 五维全高、最大化参与度 |

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
│   ├── market/                  # 行情数据：Mock/Live/CME Databento 数据源 + 技术指标 + 对抗性场景
│   ├── execution/               # 执行层：信号 + 纸上交易 + 聚合 + 风控 + 成本 + 漂移 + 辩论 + 策略
│   ├── integration/             # 外部集成：Redis 消息总线 + Telegram（信号+漂移+成本告警）
│   ├── utils/                   # 工具：配置 + 日志 + 匿名化 + 全链路日志 + TF-IDF + 知识图谱
│   └── main.py                  # 主入口
├── tests/                       # 253 个测试，覆盖全部模块
├── scripts/
│   ├── dashboard.py             # Rich 终端实时仪表盘
│   ├── backtest.py              # 规则回测
│   ├── llm_backtest.py          # LLM 真实回测（多次运行 + 一致性 + 多市况）
│   ├── generate_synthetic_data.py # 合成多市况数据（熊市/横盘/牛市）
│   ├── export_training_data.py  # 决策轨迹导出（JSONL 微调格式）
│   ├── create_agents_config.py  # 批量生成 Agent 配置
│   ├── generate_cme_data.py      # 生成合成 CME 期货 OHLCV 数据
│   └── download_cme_data.py      # 通过 Databento API 下载真实 CME 数据
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

## P4：多市场支持 — CME 期货

系统现支持 **CME 期货** 与加密货币并行，通过 `trading.yaml` 配置切换：

| 市场 | 资产 | 数据源 | 成本模式 |
|------|------|--------|---------|
| 加密货币 | BTC-PERP, ETH-PERP, SOL-PERP, ARB-PERP, DOGE-PERP | Binance REST | 滑点 + 百分比费率 + 资金费率 |
| CME 期货 | ES, NQ, CL, GC, SI, ZB | Databento API | 滑点 + 按手佣金（无资金费率） |

**CME 合约规格**（已通过 cmegroup.com 官网验证）：

| 合约 | 名称 | 乘数 | Tick Size | Tick Value |
|------|------|------|-----------|-----------|
| ES | E-mini 标普 500 | $50 | 0.25 | $12.50 |
| NQ | E-mini 纳斯达克 100 | $20 | 0.25 | $5.00 |
| CL | 原油 | $1,000 | $0.01 | $10.00 |
| GC | 黄金 | $100 | $0.10 | $10.00 |
| SI | 白银 | $5,000 | $0.005 | $25.00 |
| ZB | 美国国债 | $1,000 | 1/32 | $31.25 |

**CME 成本模型** — 无资金费率，使用按手佣金：
```yaml
# config/trading.yaml → trading.cme.costs
slippage_bps: 2                # 比加密货币低（2 vs 5 bps）
commission_per_contract: 1.25  # 单边佣金（美元/手，broker-only 估算）
enable_costs: true
```

**Prompt 市场感知**：`prompt_generator.py` 根据市场类型自动切换角色描述和维度解释（如「探索新山寨币」→「探索多样化期货合约」）。

**数据源**：
- **Mock 模式**：CSV 回放，复用 `MockDataFeed`
- **Live 模式**：`DatabentoCMEFeed` 封装 Databento SDK（需 `DATABENTO_API_KEY`）
- **合成数据**：`scripts/generate_cme_data.py` 生成 6 个合约各 2000 条 GBM 价格路径

---

## P5：CME LLM 回测修复 + 多品种对比

修复了 4 个阻塞性 Bug，使 `llm_backtest.py --market cme` 真正可用：

| Bug | 严重程度 | 修复方案 |
|-----|---------|---------|
| 成本配置路径断裂（`trading.costs` 不存在） | 严重 | 改为读取 `trading.{market_type}.costs` |
| `account.py` 完全忽略 CME 成本路径 | 严重 | 新增 `CMECostConfig` + `contract_multiplier` 到 `AgentAccount` |
| LLM 成本估算硬编码为 Claude Sonnet（$0.0135/调用） | 中等 | 按模型自动检测：DeepSeek ~$0.001，Claude ~$0.0135，GPT-4o-mini ~$0.0006 |
| 无多品种对比模式 | 功能缺失 | 新增 `--assets ES CL GC ZB` CLI 参数 |

**多品种对比** — 跨多个 CME 合约回测并汇总对比：
```bash
python scripts/llm_backtest.py --market cme --assets ES CL GC ZB --runs 2 --anonymize --max-steps 50
```
输出：每个品种每个 Agent 的 PnL/Sharpe/交易数 + 跨品种对比表。

---

## P7：32-Agent 回测深度优化

从 7 个扩展到 **32 个人格原型**（2^5 二元 SLOAN 全覆盖），并解决 32-Agent × 4 品种回测中发现的关键问题：

| 修复项 | 根因 | 严重程度 | 解决方案 |
|--------|------|---------|---------|
| **confidence=0.0 兜底** | DeepSeek 对非主流品种输出 `confidence: 0.0`（BUY/SELL 却 conf=0 的矛盾组合） | 致命 | 代码级兜底：BUY/SELL + conf=0.0 时自动提升到 0.3（低置信度但可交易） |
| **Prompt JSON 示例** | DeepSeek 需要具体的 JSON 示例才能正确输出格式 | 致命 | System Prompt 中增加完整 JSON 响应示例 |
| **Confidence 校准规则** | LLM 不知道 conf=0 + BUY 是矛盾的 | 致命 | 增加规则：「BUY/SELL 的 confidence 必须 > 0」 |
| **max_concurrent_positions 强制执行** | 字段存在但回测中从未检查 | 高 | `validate_signal()` 在 BUY 前检查 `current_positions >= max_concurrent` |
| **SELL 持仓检查** | SELL 信号通过校验后在执行时才报错 | 中 | `validate_signal()` 在无持仓时拒绝 SELL |
| **止损/止盈执行** | 持仓从未检查是否触发止损/止盈价格 | 高 | 每步决策前调用 `check_stop_loss_take_profit()` |
| **品种描述注入** | DeepSeek 不熟悉 CL/GC/ZB → 输出 conf=0 | 中 | `ASSET_DESCRIPTIONS` 字典注入 Decision Prompt |
| **持仓占比信息** | LLM 不知道可用余额百分比和已用仓位数 | 低 | 增加 `Available Balance: $X (Y%)` 和 `Positions Used: N/M` |
| **CME 主流资产扩展** | 低 O 的 Agent（16/32）被阻止交易 CL/GC/ZB | 高 | 将 `major_assets` 扩展为包含全部 5 个 CME 合约 |

**confidence=0.0 兜底逻辑**（`_backtest_helpers.py`）：
```python
# DeepSeek 常输出 conf=0.0 + BUY/SELL 的矛盾组合
# 代码级兜底：提升到 0.3（低置信度但可交易）
if action_str in ("BUY", "SELL") and confidence == 0.0:
    confidence = 0.3
```

确保全部 32 个 Agent 都能在 4 个 CME 品种（ES/CL/GC/ZB）上交易，而非之前只有 3 个高 O Agent 能交易。

---

## 三层记忆系统（FinMem 启发）

| 层级 | 名称 | 内容 | 容量 | 存储 | 检索方式 |
|------|------|------|------|------|---------|
| L1 | 工作记忆 | 最近 20 条 tick + 最近 5 次交易结果 | 20+5 | 内存 | 全量（每次决策） |
| L2 | 情节记忆 | 完整交易记录（价格、盈亏、reasoning） | 50 笔 | Redis | TF-IDF 混合检索 |
| L3 | 语义记忆 | 反思总结（自然语言） | 20 条 | Redis | 指数衰减加权 |

**记忆隔离**：每个 Agent 的记忆完全独立，互不共享，确保性格差异不被稀释。

---

## 完整部署教程（Ubuntu VPS）

本节以全新的 Ubuntu 22.04/24.04 VPS 为例，手把手教你从零开始部署到运行。

### Step 1：服务器基础环境

```bash
# 登录 VPS（替换为你的 IP 和用户名）
ssh root@your-server-ip

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y git curl wget build-essential software-properties-common
```

### Step 2：安装 Python 3.11+

```bash
# 添加 deadsnakes PPA（Ubuntu 22.04 默认是 Python 3.10，需要升级）
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# 安装 Python 3.11 和相关工具
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 验证版本
python3.11 --version
# 应输出: Python 3.11.x

# （可选）设为默认 python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

> **Ubuntu 24.04** 自带 Python 3.12，可跳过此步骤，直接用 `python3`。

### Step 3：安装 Redis

Redis 用于 Agent 间消息通信和记忆存储。

```bash
# 安装 Redis 服务器
sudo apt install -y redis-server

# 启动并设为开机自启
sudo systemctl start redis-server
sudo systemctl enable redis-server

# 验证 Redis 是否正常运行
redis-cli ping
# 应输出: PONG
```

### Step 4：克隆项目并创建虚拟环境

```bash
# 克隆项目
cd /opt
git clone https://github.com/wuyutanhongyuxin-cell/LLM_try2.git personality-trading
cd personality-trading

# 创建 Python 虚拟环境
python3.11 -m venv venv

# 激活虚拟环境（每次新开终端都需要执行）
source venv/bin/activate

# 安装项目依赖（含开发依赖）
pip install -e ".[dev]"

# 验证安装
pytest tests/ -v
# 应看到 223 passed
```

### Step 5：配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑环境变量
nano .env
```

`.env` 文件内容（根据你的需求配置）：

```bash
# ── LLM API Key（三选一即可）──

# DeepSeek（最便宜，推荐初次使用）
# 注册: https://platform.deepseek.com → API Keys → 创建
DEEPSEEK_API_KEY=sk-...

# Anthropic Claude（最强但最贵）
# 注册: https://console.anthropic.com → API Keys
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI（GPT-4o-mini 性价比不错）
# 注册: https://platform.openai.com → API Keys
OPENAI_API_KEY=sk-...

# ── 基础设施 ──

# Redis 连接（本地默认即可，不用改）
REDIS_URL=redis://localhost:6379/0

# ── Telegram 通知（可选，不用可留空）──

# 1. 在 Telegram 搜索 @BotFather → /newbot → 获取 Token
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...

# 2. 在 Telegram 搜索 @userinfobot → 获取你的 Chat ID
TELEGRAM_CHAT_ID=123456789

# ── CME 期货数据（可选，用合成数据可不填）──

# Databento 注册: https://databento.com → 获取 API Key
DATABENTO_API_KEY=db-...

# ── 日志级别 ──
LOG_LEVEL=INFO
```

> **省钱建议**：DeepSeek V3 的 API 价格约 $0.27/M input + $1.10/M output，单次回测（50 步 × 3 Agent × 3 次投票）成本约 $0.45。Claude Sonnet 同样回测约 $6.08。

### Step 6：选择 LLM 模型

编辑 `config/llm.yaml`，选择你要使用的模型：

```bash
nano config/llm.yaml
```

```yaml
llm:
  provider: "deepseek"
  model: "deepseek/deepseek-chat"    # ← 改这里切换模型
  temperature: 0.3
  max_tokens: 1024
  timeout_seconds: 30
  retry_count: 3
  retry_delay_seconds: 5
  max_calls_per_agent_per_hour: 12
  fallback_model: "deepseek/deepseek-chat"
  decision_samples: 3            # 每次决策调 3 次 LLM 投票
  consensus_threshold: 0.6       # 60% 票数才执行
  max_calls_per_minute: 20       # 全局限流（防止打爆 API）
  max_cost_per_backtest_usd: 50.0  # 回测花费上限
```

**模型选择对照表**：

| model 值 | 提供商 | 需要的 API Key | 成本/调用 | 适用场景 |
|----------|--------|---------------|----------|---------|
| `deepseek/deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` | ~$0.001 | 日常测试、大量回测 |
| `deepseek/deepseek-reasoner` | DeepSeek | `DEEPSEEK_API_KEY` | ~$0.003 | 需要更准确推理时 |
| `claude-sonnet-4-20250514` | Anthropic | `ANTHROPIC_API_KEY` | ~$0.014 | 最佳决策质量 |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` | ~$0.001 | 性价比折中 |

### Step 7：选择市场类型

编辑 `config/trading.yaml`，选择要交易的市场：

```bash
nano config/trading.yaml
```

找到 `market_type` 行：

```yaml
trading:
  market_type: "crypto"    # ← 改为 "cme" 可切换到 CME 期货市场
```

- **`"crypto"`**：加密货币永续合约（BTC-PERP、ETH-PERP 等）
- **`"cme"`**：CME 期货（ES、NQ、CL、GC、SI、ZB）

### Step 8：生成合成数据（首次运行必做）

项目自带合成数据生成脚本，不需要真实行情数据就能回测。

```bash
# 生成加密货币合成数据（如果你还没有 btc_1h_2024.csv）
# 需要提供一个种子 CSV 或者使用已有的 data/ 目录下的文件

# 生成 CME 期货合成数据（6 个品种各 2000 条 1h K 线）
python scripts/generate_cme_data.py
# 输出:
#   data/es_1h_2024.csv: 2000 行, 5200.00 → xxxx.xx
#   data/nq_1h_2024.csv: 2000 行, 18000.00 → xxxx.xx
#   data/cl_1h_2024.csv: 2000 行, 72.00 → xx.xx
#   data/gc_1h_2024.csv: 2000 行, 2400.00 → xxxx.xx
#   data/si_1h_2024.csv: 2000 行, 28.00 → xx.xx
#   data/zb_1h_2024.csv: 2000 行, 118.00 → xxx.xx
```

### Step 9：运行你的第一次回测

```bash
# 激活虚拟环境（如果还没激活）
source venv/bin/activate

# 最简单的加密货币回测（规则模式，不花钱）
python scripts/backtest.py --market crypto --csv data/btc_1h_2024.csv

# 最简单的 CME 期货回测（规则模式，不花钱）
python scripts/backtest.py --market cme --asset ES --csv data/es_1h_2024.csv
```

### Step 10：启动实时系统

```bash
# 启动多 Agent 并行决策系统
python -m src.main

# 另开一个终端，启动实时仪表盘
# ssh root@your-server-ip
# cd /opt/personality-trading && source venv/bin/activate
python scripts/dashboard.py
```

> 按 `Ctrl+C` 可优雅关闭系统（会自动保存状态）。

### （可选）用 systemd 设置后台自启

```bash
# 创建 systemd 服务文件
sudo tee /etc/systemd/system/trading-agents.service > /dev/null << 'EOF'
[Unit]
Description=Personality Trading Agents
After=redis-server.service network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/personality-trading
ExecStart=/opt/personality-trading/venv/bin/python -m src.main
Restart=on-failure
RestartSec=10
Environment="PATH=/opt/personality-trading/venv/bin:/usr/bin"

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable trading-agents
sudo systemctl start trading-agents

# 查看运行状态
sudo systemctl status trading-agents

# 查看实时日志
journalctl -u trading-agents -f
```

### （可选）用 screen/tmux 保持后台运行

如果不想用 systemd，也可以用 screen：

```bash
# 安装 screen
sudo apt install -y screen

# 创建一个 screen 会话运行系统
screen -S trading
source venv/bin/activate
python -m src.main
# 按 Ctrl+A 然后按 D 可以把 screen 放到后台

# 重新连接
screen -r trading
```

---

## CME 期货品种选择详解

### 支持的 6 个 CME 品种

| 品种代码 | 名称 | 合约乘数 | 合约价值（约） | 波动特征 | 适合新手？ |
|---------|------|---------|--------------|---------|----------|
| **ES** | E-mini 标普 500 | $50/点 | ~$295,000 | 低波动，趋势清晰 | 推荐 |
| **NQ** | E-mini 纳斯达克 100 | $20/点 | ~$410,000 | 中波动，科技股驱动 | 推荐 |
| **CL** | WTI 原油 | $1,000/桶 | ~$70,000 | 高波动，地缘政治敏感 | 进阶 |
| **GC** | 黄金 | $100/盎司 | ~$300,000 | 中低波动，避险属性 | 推荐 |
| **SI** | 白银 | $5,000/盎司 | ~$165,000 | 高波动，工业+贵金属 | 进阶 |
| **ZB** | 美国国债 30年 | $1,000/点 | ~$112,000 | 极低波动，利率驱动 | 推荐 |

### 如何选择品种

**新手入门**（推荐先跑这些）：
- `ES`（标普）：最流动、最稳定的期货合约，全球交易量第一
- `GC`（黄金）：波动适中，逻辑清晰（避险资产）
- `ZB`（国债）：波动最小，适合理解系统机制

**进阶组合**：
- `ES CL GC ZB`：经典 4 品种组合，覆盖股指+商品+贵金属+固定收益
- `ES NQ CL GC SI ZB`：全品种对比，观察不同性格 Agent 在不同品种上的差异

---

## 回测命令大全

### 1. 规则回测（不调 LLM，不花钱）

规则回测用硬编码的买入/卖出规则（基于 RSI 和价格变化），不调用任何 LLM API，适合验证系统是否正常工作。

```bash
# ── 加密货币规则回测 ──
python scripts/backtest.py --market crypto --csv data/btc_1h_2024.csv

# ── CME 期货规则回测（逐品种）──

# ES（标普 500）
python scripts/backtest.py --market cme --asset ES --csv data/es_1h_2024.csv

# CL（原油）
python scripts/backtest.py --market cme --asset CL --csv data/cl_1h_2024.csv

# GC（黄金）
python scripts/backtest.py --market cme --asset GC --csv data/gc_1h_2024.csv

# ZB（国债）
python scripts/backtest.py --market cme --asset ZB --csv data/zb_1h_2024.csv

# NQ（纳斯达克）
python scripts/backtest.py --market cme --asset NQ --csv data/nq_1h_2024.csv

# SI（白银）
python scripts/backtest.py --market cme --asset SI --csv data/si_1h_2024.csv
```

**参数说明**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--market` | 市场类型：`crypto` 或 `cme` | `--market cme` |
| `--asset` | 交易品种代码 | `--asset ES` |
| `--csv` | 历史数据 CSV 文件路径 | `--csv data/es_1h_2024.csv` |

### 2. LLM 回测（调用真实 LLM，会产生 API 费用）

LLM 回测调用真实的大模型做交易决策，结果更有意义但会消耗 API 额度。

```bash
# ── 加密货币 LLM 回测 ──

# 基础回测：3 次运行 × 3 个 Agent × 500 步
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 3 --agents 3 --anonymize

# 快速试跑（省钱版）：1 次运行 × 2 个 Agent × 30 步
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 1 --agents 2 --max-steps 30

# ── CME 期货 LLM 回测（单品种）──

# ES 标普 500 回测
python scripts/llm_backtest.py \
  --market cme \
  --asset ES \
  --csv data/es_1h_2024.csv \
  --runs 2 \
  --agents 3 \
  --anonymize \
  --max-steps 100

# CL 原油回测
python scripts/llm_backtest.py \
  --market cme \
  --asset CL \
  --csv data/cl_1h_2024.csv \
  --runs 2 \
  --agents 3 \
  --anonymize \
  --max-steps 100

# GC 黄金回测
python scripts/llm_backtest.py \
  --market cme \
  --asset GC \
  --csv data/gc_1h_2024.csv \
  --runs 2 \
  --agents 3 \
  --anonymize \
  --max-steps 100

# ZB 国债回测
python scripts/llm_backtest.py \
  --market cme \
  --asset ZB \
  --csv data/zb_1h_2024.csv \
  --runs 2 \
  --agents 3 \
  --anonymize \
  --max-steps 100
```

### 3. 多品种对比回测

同时回测多个 CME 品种，生成跨品种对比表。

```bash
# ── 经典 4 品种对比（ES + CL + GC + ZB）──
python scripts/llm_backtest.py \
  --market cme \
  --assets ES CL GC ZB \
  --csv-dir data \
  --runs 2 \
  --agents 3 \
  --anonymize \
  --max-steps 50

# ── 全 6 品种对比 ──
python scripts/llm_backtest.py \
  --market cme \
  --assets ES NQ CL GC SI ZB \
  --csv-dir data \
  --runs 1 \
  --agents 3 \
  --anonymize \
  --max-steps 30

# ── 只对比股指（ES vs NQ）──
python scripts/llm_backtest.py \
  --market cme \
  --assets ES NQ \
  --csv-dir data \
  --runs 3 \
  --agents 3 \
  --anonymize \
  --max-steps 100

# ── 只对比商品（CL + GC + SI）──
python scripts/llm_backtest.py \
  --market cme \
  --assets CL GC SI \
  --csv-dir data \
  --runs 2 \
  --agents 3 \
  --anonymize \
  --max-steps 80
```

### 4. 多市况压力测试

```bash
# 先生成不同市况的合成数据
python scripts/generate_synthetic_data.py --csv data/btc_1h_2024.csv --output data/

# 然后用 --multi-market 参数跨市况对比
python scripts/llm_backtest.py \
  --csv data/btc_bull.csv \
  --runs 3 \
  --multi-market \
  --anonymize \
  --max-steps 100
```

### LLM 回测参数完整说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv` | 字符串 | **必填** | 历史数据 CSV 文件路径 |
| `--runs` | 整数 | 3 | 重复运行次数（用于收集决策一致性数据） |
| `--agents` | 整数 | 32 | 使用前 N 个预定义人格原型（共 32 个） |
| `--anonymize` | 开关 | 关闭 | 启用资产匿名化（BTC-PERP → ASSET_A），防止 LLM 回忆历史 |
| `--max-steps` | 整数 | 500 | 最大回测步数（每步 = 1 根 K 线） |
| `--market` | 选项 | crypto | 市场类型：`crypto` 或 `cme` |
| `--asset` | 字符串 | 自动 | 单品种交易代码（如 `ES`、`BTC-PERP`） |
| `--assets` | 列表 | 无 | 多品种对比模式，空格分隔（如 `ES CL GC ZB`） |
| `--csv-dir` | 字符串 | data | 多品种模式下 CSV 文件所在目录 |
| `--multi-market` | 开关 | 关闭 | 跨市况对比（需先生成合成数据） |

### 费用估算参考

| 场景 | 参数 | DeepSeek 费用 | Claude 费用 | GPT-4o-mini 费用 |
|------|------|-------------|------------|-----------------|
| 快速试跑 | `--runs 1 --agents 2 --max-steps 30` | ~$0.18 | ~$2.43 | ~$0.11 |
| 标准回测 | `--runs 3 --agents 3 --max-steps 100` | ~$2.70 | ~$36.45 | ~$1.62 |
| 4 品种对比 | `--assets ES CL GC ZB --runs 2 --max-steps 50` | ~$3.60 | ~$48.60 | ~$2.16 |
| 全品种深度 | `--assets ES NQ CL GC SI ZB --runs 3 --max-steps 100` | ~$16.20 | ~$218.70 | ~$9.72 |

> 计算公式：`费用 = runs × agents × max_steps × decision_samples(3) × 单次调用成本`
> 系统有 `max_cost_per_backtest_usd` 硬上限保护（默认 $50），超限自动停止。

---

## 配置详解

### 交易成本配置（`config/trading.yaml`）

**加密货币成本**（位于 `trading.crypto.costs`）：
```yaml
trading:
  crypto:
    costs:
      slippage_bps: 5              # 滑点 5 bps = 0.05%
      taker_fee_rate: 0.0004       # Taker 手续费 0.04%（吃单）
      maker_fee_rate: 0.0002       # Maker 手续费 0.02%（挂单）
      funding_rate_8h: 0.00015     # 8h 资金费率 0.015%
      enable_costs: true           # false 可关闭（A/B 对比实验用）
```

**CME 期货成本**（位于 `trading.cme.costs`）：
```yaml
trading:
  cme:
    costs:
      slippage_bps: 2              # 滑点 2 bps（比加密货币低，CME 流动性好）
      commission_per_contract: 1.25 # 单边佣金 $1.25/手（broker-only 估算）
      enable_costs: true
    contracts:                     # 各品种合约规格
      ES: { multiplier: 50, tick_size: 0.25, tick_value: 12.50 }
      NQ: { multiplier: 20, tick_size: 0.25, tick_value: 5.00 }
      CL: { multiplier: 1000, tick_size: 0.01, tick_value: 10.00 }
      GC: { multiplier: 100, tick_size: 0.10, tick_value: 10.00 }
      SI: { multiplier: 5000, tick_size: 0.005, tick_value: 25.00 }
      ZB: { multiplier: 1000, tick_size: 0.03125, tick_value: 31.25 }
```

### 通用交易配置

```yaml
trading:
  market_type: "crypto"          # 全局市场类型切换："crypto" | "cme"

  data_feed:
    type: "mock"                 # "mock"（CSV 回放）| "live"（实时行情）
    interval_seconds: 60         # 行情轮询间隔（秒）

  aggregator:
    mode: "independent"          # "independent"（独立对比）| "voting"（投票集成）
    signal_window_seconds: 120   # 信号收集时间窗口
    enable_debate: false         # true 启用 Bull/Bear 辩论（仅 voting 模式）

  anonymize: false               # 资产匿名化（回测建议开 true）

  risk:
    global_max_drawdown_pct: 25  # 全局最大回撤限制 25%
    global_max_daily_loss_pct: 10 # 全局单日最大亏损 10%
```

### LLM 配置（`config/llm.yaml`）

```yaml
llm:
  provider: "deepseek"
  model: "deepseek/deepseek-chat"
  temperature: 0.3               # 越低越确定（0.0-1.0）
  max_tokens: 1024               # 单次响应最大 token 数
  timeout_seconds: 30            # 超时时间
  retry_count: 3                 # 失败重试次数
  retry_delay_seconds: 5         # 重试间隔
  max_calls_per_agent_per_hour: 12  # 每 Agent 每小时最多调用 12 次
  fallback_model: "deepseek/deepseek-chat"  # 主模型失败后的备选模型
  decision_samples: 3            # 每次决策调 LLM 次数（1=不投票，3=三票制）
  consensus_threshold: 0.6       # 多数票占比阈值（低于 60% 默认 HOLD）
  max_calls_per_minute: 20       # 全局每分钟限流
  max_cost_per_backtest_usd: 50.0  # 单次回测花费硬上限（美元）
```

### Agent 人格配置（`config/agents.yaml`）

```yaml
agents:
  # 使用预定义原型（32 个可选，基于 2^5 二元 OCEAN 组合）
  - id: "agent_calm_innovator"
    preset: "冷静创新型"           # ★ HHLLL O=90 C=80 E=25 A=20 N=10
    initial_capital: 10000

  - id: "agent_conservative"
    preset: "保守焦虑型"           # ★ LHLHH O=15 C=85 E=20 A=70 N=90
    initial_capital: 10000

  - id: "agent_iron_defense"
    preset: "铁壁防守型"           # LHLLL O=20 C=80 E=20 A=20 N=20
    initial_capital: 10000

  # 也可以自定义 OCEAN 参数：
  - id: "agent_custom_1"
    custom:
      name: "自定义策略型"
      openness: 70            # 0-100
      conscientiousness: 60
      extraversion: 40
      agreeableness: 30
      neuroticism: 55
    initial_capital: 10000
```

**可用预定义原型**：共 32 个，基于 2^5 二元 OCEAN 穷举。4 个经典原型：`冷静创新型`、`保守焦虑型`、`激进冒险型`、`情绪追涨型`；28 个扩展原型详见上方完整表格。

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
| CME 数据 | `databento` | Databento API 获取 CME 期货 OHLCV |
| 测试 | pytest + pytest-asyncio | 253 个测试，全模块覆盖 |

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
- [x] OCEAN 人格模型 + 32 个原型（2^5 二元 SLOAN 全覆盖）
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

### P4（完成）：多市场支持 — CME 期货
- [x] `config/trading.yaml` 多市场结构（market_type: crypto | cme）
- [x] CME 合约规格（ES/NQ/CL/GC/SI/ZB 乘数，已通过 cmegroup.com 验证）
- [x] `databento_feed.py` — CME 数据源（Live via Databento + Mock 回退）
- [x] `cost_model.py` — CMECostConfig 按手佣金（无资金费率）
- [x] `prompt_generator.py` 市场感知角色和维度描述
- [x] `market_knowledge.json` 扩展 CME 因果关系

### P5（完成）：CME LLM 回测修复 + 4 品种对比
- [x] 修复成本配置路径（读取市场专属配置段）
- [x] `account.py` + `paper_trader.py` — 完整 CME 成本路径（commission_per_contract）
- [x] 按模型自动估算 LLM 成本（DeepSeek $0.001 / Claude $0.0135 / GPT-4o-mini $0.0006）
- [x] `--assets ES CL GC ZB` 多品种对比模式 + 跨品种对比表
- [x] `databento_feed.py` ImportError 安全降级

### P6（完成）：LLM 回测优化
- [x] 可配置置信度缩放（`backtest_confidence_scale: 0.6`）— 修复零交易 Agent
- [x] 诊断日志：`REJECTED` vs `HOLD` 区分 + 每个拒绝路径的 debug 日志
- [x] 空 LLM 响应自动重试（最多 3 次）
- [x] 单次运行一致性计算（`--runs 1` 不再跳过）
- [x] 未平仓头寸显示（`"0+1open"` 格式）+ Actions 统计列
- [x] 初始资金提升到 $5M（CME 期货合理规模）
- [x] Logger import 修复 — `LOG_LEVEL=DEBUG` 对回测脚本生效
- [x] System Prompt 主动交易指导 + confidence 校准
- [x] Decision Prompt 技术指标注入 — RSI(14)、SMA(20)、MACD
- [x] `prompt_constants.py` 从 `prompt_generator.py` 拆分（文件行数合规）

### P7（完成）：32-Agent 回测深度优化
- [x] 7→32 人格原型（2^5 二元 SLOAN 全覆盖）
- [x] confidence=0.0 代码级兜底（DeepSeek BUY/SELL + conf=0 → 自动提升到 0.3）
- [x] Prompt JSON 示例 + confidence 校准规则
- [x] `max_concurrent_positions` 在 `validate_signal()` 中强制执行
- [x] SELL 持仓检查（无持仓时拒绝 SELL）
- [x] 止损/止盈在回测循环中执行
- [x] 品种描述注入（`ASSET_DESCRIPTIONS` 注入 Decision Prompt）
- [x] 持仓占比信息增强（余额百分比、已用仓位数）
- [x] CME `major_assets` 扩展为 5 个合约（全部 Agent 可交易全部品种）

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
