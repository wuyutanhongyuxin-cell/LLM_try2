# CLAUDE.md — Personality-Conditioned Multi-Agent Crypto Trading System

> 把这个文件放在项目根目录。Claude Code 每次启动自动读取。

---

## [项目专属区域]

### 项目名称
personality-trading-agents

### 一句话描述
基于 Big Five (OCEAN) 人格模型的多Agent加密货币纸上交易系统——每个Agent被赋予不同性格参数，性格决定交易风格，多Agent并行生成信号并对比绩效。

### 技术栈

| 类别 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.11+ | 全项目统一 |
| 异步 | asyncio + aiohttp | 所有IO必须async |
| LLM | litellm (`acompletion`) | 统一接口，禁止直接调SDK |
| 消息队列 | Redis pub/sub | 已有基础设施 |
| Telegram | aiogram 3.x | 已有基础设施 |
| 数据校验 | Pydantic v2 | 所有数据结构必须用Model |
| 配置 | YAML | `config/` 目录 |
| 日志 | loguru | 统一日志 |
| 测试 | pytest + pytest-asyncio | 覆盖核心逻辑 |
| 金额计算 | `decimal.Decimal` | 禁止float算钱 |

**禁止引入的依赖**：pandas、numpy、django、flask、sqlalchemy。保持轻量。

### 项目结构

```
personality-trading-agents/
├── CLAUDE.md                          # 本文件
├── todo.md                            # 进度跟踪
├── pyproject.toml                     # 依赖声明
├── .env.example                       # 环境变量模板
├── config/
│   ├── agents.yaml                    # Agent人格配置（OCEAN参数）
│   ├── trading.yaml                   # 交易参数（交易对、数据源、风控）
│   └── llm.yaml                       # LLM配置（provider、model、温度）
├── src/
│   ├── __init__.py
│   ├── main.py                        # 入口：启动所有Agent + 聚合器
│   ├── personality/                   # 人格引擎
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── ocean_model.py             # OCEAN参数定义 + 预定义原型（~80行）
│   │   ├── prompt_generator.py        # OCEAN → System/Decision Prompt（~120行）
│   │   └── trait_to_constraint.py     # OCEAN → 硬编码交易约束（~80行）
│   ├── agent/                         # Agent核心
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Agent基类：生命周期（~60行）
│   │   ├── trading_agent.py           # 交易Agent：多采样+匿名化（~200行）
│   │   ├── multi_sample.py            # 多采样投票决策（NEW ~45行）
│   │   ├── memory.py                  # 三层记忆+相关性检索+衰减（~160行）
│   │   └── reflection.py             # 交易反思模块（~80行）
│   ├── market/                        # 行情数据
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── data_feed.py               # 数据源抽象 + Mock/Live实现（~150行）
│   │   └── indicators.py              # 技术指标：RSI/SMA/MACD（~100行）
│   ├── execution/                     # 执行层
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── signal.py                  # TradeSignal+prompt_hash（~50行）
│   │   ├── cost_model.py              # 交易成本：滑点+手续费+funding（NEW ~90行）
│   │   ├── consistency_monitor.py     # 行为漂移检测：三级KL阈值（NEW ~120行）
│   │   ├── aggregator.py              # 多Agent信号聚合（~100行）
│   │   ├── risk_manager.py            # 全局+Agent级风控（~100行）
│   │   └── paper_trader.py            # 纸上交易执行器+成本（~120行）
│   ├── integration/                   # 外部集成
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── redis_bus.py               # Redis pub/sub封装（~60行）
│   │   └── telegram_notifier.py       # Telegram推送（~100行）
│   └── utils/                         # 工具
│       ├── __init__.py
│       ├── logger.py                  # loguru配置（~30行）
│       ├── config_loader.py           # YAML加载器（~40行）
│       ├── anonymizer.py             # 资产匿名化防look-ahead bias（NEW ~60行）
│       └── trade_logger.py           # 全链路交易日志（NEW ~60行）
├── tests/
│   ├── test_personality/
│   │   ├── test_ocean_model.py
│   │   ├── test_prompt_generator.py
│   │   └── test_trait_to_constraint.py
│   ├── test_agent/
│   │   ├── test_trading_agent.py
│   │   └── test_memory.py
│   ├── test_execution/
│   │   ├── test_aggregator.py
│   │   ├── test_paper_trader.py
│   │   ├── test_cost_model.py         # NEW
│   │   └── test_consistency_monitor.py # NEW
│   └── test_utils/
│       └── test_anonymizer.py          # NEW
├── scripts/
│   ├── llm_backtest.py                # LLM真实回测+一致性报告（NEW）
    ├── backtest.py                    # 历史回测
    ├── create_agents_config.py        # 批量生成Agent配置
    └── dashboard.py                   # Rich终端仪表盘
```

### 当前阶段

Phase 1：纸上交易验证。详见 `todo.md`。

---

## 开发者背景

我不是专业开发者，使用 Claude Code 辅助编程。请：
- 代码加中文注释，关键逻辑额外解释
- 遇到复杂问题先给方案让我确认，不要直接大改
- 报错时解释原因 + 修复方案，不要只贴代码
- 优先最简实现，不要过度工程化

---

## 领域知识（必读）

### Big Five 人格模型

本项目的核心概念。5个维度，每个 0-100 连续分数：

| 维度 | 缩写 | 高分交易含义 | 低分交易含义 |
|------|------|-------------|-------------|
| Openness | O | 探索新币种/新策略/高波动资产 | 只做BTC/ETH，保守策略 |
| Conscientiousness | C | 严格止损/仓位管理/纪律执行 | 冲动交易，忽视风控 |
| Extraversion | E | 追涨杀跌，跟随市场情绪 | 逆向交易，独立判断 |
| Agreeableness | A | 从众/herding behavior | 对抗市场共识，做空倾向 |
| Neuroticism | N | 极度厌恶损失，止损极紧，频繁割肉 | 情绪稳定，能hold住回撤 |

**关键设计原则：LLM只负责"建议"，硬约束由代码强制执行。** LLM输出任何超出constraints的值都会被clip到合法范围。

### OCEAN → 交易约束映射公式

这些公式是硬编码的，不可被LLM覆盖：

```python
max_position_pct     = clip(5 + (100 - N) * 0.25, 5, 30)
stop_loss_pct        = clip(1 + (100 - N) * 0.14, 1, 15)
max_drawdown_pct     = clip(2 + (100 - N) * 0.18, 2, 20)
max_concurrent_pos   = clip(1 + O // 20, 1, 6)
use_sentiment_data   = E > 50
momentum_weight      = E / 100.0
contrarian_weight    = (100 - A) / 100.0
require_stop_loss    = C > 50
min_confidence       = clip(C / 100.0 * 0.8, 0.2, 0.8)

rebalance_interval:
    N > 70 → 300秒 (5分钟)
    N > 40 → 3600秒 (1小时)
    N <= 40 → 86400秒 (1天)

allowed_assets:
    O > 60 → 全部币种
    O <= 60 → 仅主流 (BTC/ETH)
```

---

## 实现顺序（严格按此，每步测试通过再进下一步）

### Step 1：基础设施
1. `pyproject.toml` — 依赖声明
2. `utils/config_loader.py` — YAML加载
3. `utils/logger.py` — loguru配置
4. 所有 `config/*.yaml`
5. `.env.example`

### Step 2：人格引擎
6. `personality/ocean_model.py` — OceanProfile + 7个预定义原型
7. `personality/trait_to_constraint.py` — 映射公式实现
8. `personality/prompt_generator.py` — Prompt生成
9. `tests/test_personality/` — **全部通过后再继续**

### Step 3：数据层
10. `market/data_feed.py` — MockDataFeed（从CSV） + LiveDataFeed（Binance REST）
11. `market/indicators.py` — RSI / SMA / MACD
12. `execution/signal.py` — TradeSignal 数据结构
13. `integration/redis_bus.py` — Redis封装

### Step 4：Agent核心
14. `agent/memory.py` — 三层记忆
15. `agent/base_agent.py` — 基类
16. `agent/trading_agent.py` — 完整决策循环
17. `tests/test_agent/` — **用MockDataFeed + 固定LLM响应测试**

### Step 5：执行层
18. `execution/paper_trader.py` — 纸上交易
19. `execution/aggregator.py` — 信号聚合
20. `execution/risk_manager.py` — 全局风控
21. `tests/test_execution/`

### Step 6：集成层
22. `integration/telegram_notifier.py`
23. `agent/reflection.py`
24. `src/main.py` — 主入口

### Step 7：工具
25. `scripts/dashboard.py` — Rich仪表盘
26. `scripts/create_agents_config.py`
27. `scripts/backtest.py`

---

## 模块详细规范

### personality/ocean_model.py

```python
from pydantic import BaseModel, Field

class OceanProfile(BaseModel):
    """Big Five人格参数，每个维度0-100"""
    name: str = Field(..., description="人格原型名称")
    openness: int = Field(..., ge=0, le=100)
    conscientiousness: int = Field(..., ge=0, le=100)
    extraversion: int = Field(..., ge=0, le=100)
    agreeableness: int = Field(..., ge=0, le=100)
    neuroticism: int = Field(..., ge=0, le=100)

# 必须实现以下7个预定义原型：
PRESET_PROFILES = {
    "冷静创新型":  (O=90, C=80, E=25, A=20, N=10),
    "保守焦虑型":  (O=15, C=85, E=20, A=70, N=90),
    "激进冒险型":  (O=85, C=20, E=80, A=15, N=10),
    "纪律动量型":  (O=50, C=90, E=75, A=50, N=30),
    "逆向价值型":  (O=60, C=75, E=10, A=10, N=25),
    "平衡中庸型":  (O=50, C=50, E=50, A=50, N=50),
    "情绪追涨型":  (O=70, C=15, E=90, A=80, N=75),
}
```

### personality/prompt_generator.py

生成两种Prompt：

**System Prompt**（每个Agent初始化时生成一次）：
1. 角色设定：你是加密货币交易员，具有以下性格...
2. 五维描述：逐维度描述分数 + 交易行为含义
3. 硬约束注入：列出TradingConstraints全部数值
4. 输出格式：强制JSON
5. 禁止事项：禁止编造数据/超出约束/输出非JSON

**Decision Prompt**（每次决策时生成）：
- 输入：行情数据 + 持仓 + 记忆上下文 + 总资产
- 要求输出JSON：
```json
{
    "action": "BUY|SELL|HOLD",
    "asset": "BTC-PERP",
    "size_pct": 15.0,
    "entry_price": 67200.0,
    "stop_loss_price": 64000.0,
    "take_profit_price": 72000.0,
    "confidence": 0.75,
    "reasoning": "...",
    "personality_influence": "N维度主导：高神经质导致..."
}
```

**Prompt全部用英文**（LLM英文推理更准），日志和通知用中文。

### agent/memory.py

三层记忆，模仿FinMem：

| 层 | 名称 | 内容 | 容量 | 存储 | 触发 |
|---|------|------|------|------|------|
| L1 | Working | 最近20条tick + 最近5次交易结果 | 20条 | 内存 | 每次决策 |
| L2 | Episodic | 每笔交易完整记录（价格/盈亏/reasoning） | 50笔 | Redis | 每次交易 |
| L3 | Semantic | 反思总结（自然语言） | 20条 | Redis | 每10笔交易 |

提供 `get_context_for_decision() -> str` 方法，从三层提取上下文，拼成字符串插入decision prompt。

### agent/trading_agent.py

每个Agent是独立的 asyncio Task，主循环：

```
loop:
    1. await asyncio.sleep(rebalance_interval)
    2. snapshot = await market_feed.get_latest(asset)
    3. context = memory.get_context_for_decision()
    4. prompt = generate_decision_prompt(snapshot, positions, context, value)
    5. response = await litellm.acompletion(model, messages=[system, user])
    6. signal = parse_and_validate(response, constraints)
    7. if signal and signal.confidence >= min_confidence:
           publish to Redis "agent_signals"
           update memory L1 + L2
    8. if trade_count % 10 == 0:
           trigger reflection → update memory L3
```

**关键**：Step 6 的 `parse_and_validate` 必须：
- 解析JSON，失败则跳过本轮
- 校验所有字段是否在constraints范围内
- 超限的字段clip到合法值
- 买入非允许资产 → 返回None

### execution/paper_trader.py

每个Agent一个虚拟账户，跟踪：
- 当前持仓 / 已实现PnL / 未实现PnL / 总资产
- Sharpe Ratio（滚动） / 最大回撤 / 胜率 / 盈亏比

提供两个关键接口：
- `get_agent_stats(agent_id) -> dict`
- `get_leaderboard() -> list[dict]` — 按Sharpe排序

行情更新时检查止损/止盈是否触发。

### execution/aggregator.py

两种模式（`config/trading.yaml` 中配置）：

| 模式 | 说明 | 用途 |
|------|------|------|
| `independent` | 各Agent信号直接执行，独立PnL | 对比实验 |
| `voting` | 按 confidence × historical_sharpe 加权投票 | 集成决策 |

### integration/telegram_notifier.py

推送格式：

```
🧠 冷静创新型 (O90/C80/E25/A20/N10)
📊 BUY BTC-PERP @ $67,200
💰 Size: 25% | SL: $64,000 | TP: $72,000
🎯 Confidence: 0.85
💭 链上数据显示鲸鱼积累，技术面RSI超卖
🔑 主导: O维度—愿意在回调中建仓新头寸

📈 Daily Report - 2026-03-17
| # | Agent      | PnL    | Sharpe | MaxDD  | Trades |
|---|-----------|--------|--------|--------|--------|
| 1 | 冷静创新型 | +$320  | 1.85   | -3.2%  | 5      |
| 2 | 逆向价值型 | +$180  | 1.42   | -2.1%  | 3      |
| 3 | 激进冒险型 | -$450  | -0.32  | -12.5% | 12     |
```

### scripts/dashboard.py

用 `rich.live` 实现终端实时仪表盘：
- 顶部：系统状态（运行时长、活跃Agent数、总信号数）
- 中部：Agent状态表格（OCEAN / 持仓 / PnL / Sharpe / 最新信号）
- 底部：最近5条信号滚动日志
- 刷新：每2秒从Redis拉取

---

## 配置文件规范

### config/agents.yaml

```yaml
agents:
  - id: "agent_calm_innovator"
    preset: "冷静创新型"
    initial_capital: 10000

  - id: "agent_conservative"
    preset: "保守焦虑型"
    initial_capital: 10000

  - id: "agent_aggressive"
    preset: "激进冒险型"
    initial_capital: 10000

  # 也支持自定义OCEAN参数：
  # - id: "agent_custom"
  #   custom:
  #     name: "自定义型"
  #     openness: 70
  #     conscientiousness: 60
  #     extraversion: 40
  #     agreeableness: 30
  #     neuroticism: 55
  #   initial_capital: 10000
```

### config/trading.yaml

```yaml
trading:
  assets:
    major: ["BTC-PERP", "ETH-PERP"]
    all: ["BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "DOGE-PERP"]
  data_feed:
    type: "mock"                  # "mock" | "live"
    source: "binance"
    interval_seconds: 60
    mock_csv_path: "data/btc_1h_2024.csv"
  aggregator:
    mode: "independent"           # "independent" | "voting"
    signal_window_seconds: 120
  risk:
    global_max_drawdown_pct: 25
    global_max_daily_loss_pct: 10
```

### config/llm.yaml

```yaml
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  temperature: 0.3
  max_tokens: 1024
  timeout_seconds: 30
  retry_count: 3
  retry_delay_seconds: 5
  max_calls_per_agent_per_hour: 12
  fallback_model: "gpt-4o-mini"
```

---

## 环境变量（.env.example）

```bash
# 必须
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# 可选
OPENAI_API_KEY=sk-...
BINANCE_BASE_URL=https://api.binance.com
LOG_LEVEL=INFO
```

---

## 验收标准

1. `pytest tests/ -v` 全部通过
2. `python -m src.main` 能启动3个Agent并行运行
3. MockDataFeed回测能生成各Agent的PnL对比
4. Telegram正确推送信号和日报
5. 不同OCEAN参数的Agent表现出不同行为：
   - 高N Agent止损频率 > 低N Agent
   - 高O Agent交易币种范围 > 低O Agent
   - 高E Agent追涨比例 > 低E Agent
6. 3个Agent × 60秒间隔 → 每小时LLM调用 ≤ 36次
7. `scripts/llm_backtest.py --runs 3 --agents 3` 能完成并输出一致性报告
8. 不同 OCEAN Agent 的 action agreement rate 应有差异（高 C > 低 C）
9. 开启交易成本后，总 PnL 低于关闭成本时的 PnL（验证成本模型生效）
10. 匿名化开关切换前后 LLM 不会拒绝响应
11. 行为漂移检测能在人工注入异常后触发告警

---

## 绝对禁止

1. **禁止**用MBTI — 只用Big Five连续维度
2. **禁止**LLM直接控制金额 — 只输出百分比，代码根据constraints算金额
3. **禁止**Agent间共享记忆 — 完全隔离
4. **禁止**自动修改OCEAN参数 — Phase 1人格固定
5. **禁止**连接真实交易所 — Phase 1只有paper trading
6. **禁止**用pandas/numpy — 纯Python + Pydantic
7. **禁止**回测不计算交易成本 — 必须包含滑点和手续费
8. **禁止**LLM回测用规则替代LLM — `llm_backtest.py` 必须调真实 LLM
9. **禁止**一致性度量只跑1次 — 至少3次运行才有统计意义
10. **禁止**修改 `trait_to_constraint.py` 映射公式 — 核心竞争力

---

## 上下文管理规范（核心）

### 1. 文件行数硬限制

| 文件类型 | 最大行数 | 超限动作 |
|----------|----------|----------|
| 单个源代码文件 | **200 行** | 立即拆分为多个文件 |
| 单个模块（目录内所有文件） | **2000 行** | 拆分为子模块 |
| 测试文件 | **300 行** | 按功能拆分测试文件 |
| 配置文件 | **100 行** | 拆分为多个配置文件 |

**每次创建或修改文件后，检查行数。接近限制时主动提醒我。**

### 2. 每个目录必须有 README.md

当一个目录下有 3 个以上文件时，创建 `README.md`：
```markdown
# 目录名

## 用途
一句话说明这个目录做什么。

## 文件清单
- `xxx.py` — 做什么（~行数）

## 依赖关系
- 本目录依赖：xxx 模块
- 被以下模块依赖：yyy
```

### 3. 定期清理（每 2-3 天执行一次）

当我说 **"清理一下"** 时，执行：

```
📊 项目健康度检查
━━━━━━━━━━━━━━━━━━━━━━
✅ 文件行数：全部 < 200 行（或列出超限文件）
✅ 死代码：未发现（或列出）
⚠️ TODO 数量：X 个（列出）
✅ 临时脚本：无（或列出）
⚠️ 描述同步：CLAUDE.md 需更新（列出差异）
✅ 依赖：全部在使用
━━━━━━━━━━━━━━━━━━━━━━
建议操作：（列出）
```

---

## Sub-Agent 并行调度规则

### 什么时候并行

**并行派遣**（所有条件满足时）：
- 3+ 个不相关任务
- 不操作同一个文件
- 无输入输出依赖

**顺序派遣**（任一条件触发时）：
- B 需要 A 的输出
- 操作同一文件
- 范围不明确

### Sub-Agent 调用要求

每次派遣 sub-agent 必须指明：
1. 操作哪些文件（写）
2. 读取哪些文件（只读）
3. 完成标准
4. 不许碰哪些文件

---

## 编码规范

### 错误处理
- 所有外部调用（LLM API、Redis、行情API）必须 try-except
- LLM调用失败 → 重试3次 → 跳过本轮决策
- JSON解析失败 → 记录原始响应到日志 → 跳过本轮
- Redis断连 → 自动重连，期间Agent暂停
- 行情异常 → 跳过本轮，不做决策

### 函数设计
- 单个函数不超过 30 行（超过就拆）
- 函数名动词开头：`generate_prompt()`, `parse_signal()`, `validate_constraints()`
- 每个函数有 docstring，说明输入输出和异常

### 类型注解
- **覆盖率100%**：所有函数参数和返回值必须有类型注解
- 所有Pydantic model必须有 `Field(description=...)`

### 依赖管理
- 不要自行引入新依赖，需要时先问我
- 新增依赖立即更新 `pyproject.toml`

### 配置管理
- 敏感信息放 `.env`，通过 `os.environ` 读取
- 非敏感配置放 `config/*.yaml`
- 绝不硬编码密钥或URL

---

## Git 规范

### Commit 格式
```
<类型>: <一句话描述>

类型：feat | fix | refactor | docs | chore | test
```

### 每次 commit 前
- 确认没有 .env / __pycache__ / .cache/
- 确认 `pytest tests/ -v` 通过

---

## 沟通规范

### 当你不确定时
- 直接说不确定，不要编造
- 给 2-3 个方案让我选，标明优缺点

### 当任务太大时
- 先给拆分计划（参考上面的实现顺序），让我确认
- 每完成一步告诉我进度

### 当代码出问题时
1. 一句话说是什么问题
2. 为什么出了这个问题
3. 修复方案
4. 不要默默改一堆东西

### 关键词速查

| 我说 | 你做 |
|------|------|
| "清理一下" | 执行项目健康度检查 |
| "拆一下" | 检查行数，给拆分方案 |
| "健康检查" | 完整检查 |
| "现在到哪了" | 总结进度，参考 todo.md |
| "省着点" | 简短回复，不重复整文件 |
| "全力跑" | 并行、可大改、不用每步确认 |

---

## 性能优化规范

### Token 节省
1. 修改文件只输出变更部分，用 `// ... existing code ...`
2. 长文件只输出相关函数
3. 简单问题简短回答

### 上下文保鲜
1. 超20轮 → 建议 `/compact`
2. 切换模块 → 建议新 session
3. 探索代码 → 用 sub-agent
4. 超30轮 / 质量下降 → 强烈建议新 session

---

## 新模块 Checklist

每次新建模块确保：
- [ ] 目录级 `README.md`
- [ ] 每个文件有 docstring + 中文注释
- [ ] 行数全部 < 200
- [ ] 类型注解100%
- [ ] 更新 CLAUDE.md 项目结构
- [ ] 更新 todo.md

## 新功能 Checklist

- [ ] 有错误处理（try-except + 友好降级）
- [ ] 不引入新依赖（或已获批准）
- [ ] 文件行数未超限
- [ ] 有对应的测试
- [ ] 能独立运行不报错
