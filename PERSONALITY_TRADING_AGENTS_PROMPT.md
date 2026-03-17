# CLAUDE.md — Personality-Conditioned Multi-Agent Crypto Trading System

## 项目概述

构建一个基于 Big Five (OCEAN) 人格模型的多Agent加密货币交易系统。每个Agent被赋予不同的Big Five人格参数，这些参数决定了其交易风格、风险偏好、仓位管理和决策逻辑。多个Agent并行运行，各自独立生成交易信号，最终通过集成层聚合决策。

**核心理念**：用心理学人格理论作为策略多样性的先验约束，而非手动调参——每个Agent的交易行为由其"性格"自然涌现。

---

## 技术栈要求

- **语言**: Python 3.11+
- **异步框架**: asyncio + aiohttp
- **LLM接口**: litellm（已有基础设施，支持Claude/GPT/本地模型统一调用）
- **消息队列**: Redis pub/sub（已有基础设施）
- **Telegram通知**: aiogram 3.x（已有基础设施）
- **数据格式**: Pydantic v2 做数据校验
- **配置管理**: YAML配置文件
- **日志**: loguru
- **测试**: pytest + pytest-asyncio

---

## 目录结构

严格按照以下结构创建项目：

```
personality-trading-agents/
├── CLAUDE.md                          # 本文件
├── pyproject.toml                     # 项目依赖
├── config/
│   ├── agents.yaml                    # Agent人格配置（所有Agent的OCEAN参数）
│   ├── trading.yaml                   # 交易参数（交易对、交易所、全局风控）
│   └── llm.yaml                       # LLM配置（provider、model、temperature等）
├── src/
│   ├── __init__.py
│   ├── main.py                        # 入口：启动所有Agent + 信号聚合器
│   ├── personality/
│   │   ├── __init__.py
│   │   ├── ocean_model.py             # Big Five人格参数定义与校验
│   │   ├── prompt_generator.py        # OCEAN参数 → System Prompt 生成器
│   │   └── trait_to_constraint.py     # OCEAN参数 → 硬编码交易约束映射
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Agent基类：生命周期管理
│   │   ├── trading_agent.py           # 具体交易Agent实现
│   │   ├── memory.py                  # 三层记忆系统（短期/中期/长期）
│   │   └── reflection.py             # Agent交易后反思模块
│   ├── market/
│   │   ├── __init__.py
│   │   ├── data_feed.py               # 行情数据源抽象层（WebSocket）
│   │   ├── indicators.py              # 技术指标计算
│   │   └── sentiment.py               # 市场情绪数据获取（可选）
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── signal.py                  # 交易信号数据结构
│   │   ├── aggregator.py              # 多Agent信号聚合器
│   │   ├── risk_manager.py            # 全局风控（跨Agent层面）
│   │   └── paper_trader.py            # 纸上交易执行器
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── redis_bus.py               # Redis pub/sub 消息总线
│   │   └── telegram_notifier.py       # Telegram 通知推送
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # loguru配置
│       └── config_loader.py           # YAML配置加载器
├── tests/
│   ├── test_personality/
│   │   ├── test_ocean_model.py
│   │   ├── test_prompt_generator.py
│   │   └── test_trait_to_constraint.py
│   ├── test_agent/
│   │   ├── test_trading_agent.py
│   │   └── test_memory.py
│   └── test_execution/
│       ├── test_aggregator.py
│       └── test_paper_trader.py
└── scripts/
    ├── backtest.py                    # 历史回测脚本
    ├── create_agents_config.py        # 批量生成Agent人格配置
    └── dashboard.py                   # Rich终端仪表盘（实时显示各Agent状态）
```

---

## 模块详细规范

### 1. `personality/ocean_model.py`

```python
"""
Big Five (OCEAN) 人格模型定义。

每个维度的分数范围为 0-100 的连续整数。
提供预定义的人格原型和自定义人格创建。
"""
from pydantic import BaseModel, Field, field_validator

class OceanProfile(BaseModel):
    """Big Five人格参数"""
    name: str = Field(..., description="人格原型名称，如'冷静创新型'")
    openness: int = Field(..., ge=0, le=100, description="开放性: 高=探索新策略新币种, 低=只做主流")
    conscientiousness: int = Field(..., ge=0, le=100, description="尽责性: 高=严格风控纪律, 低=冲动交易")
    extraversion: int = Field(..., ge=0, le=100, description="外向性: 高=追随市场情绪, 低=逆向独立判断")
    agreeableness: int = Field(..., ge=0, le=100, description="宜人性: 高=从众跟风, 低=对抗市场共识")
    neuroticism: int = Field(..., ge=0, le=100, description="神经质: 高=极度厌恶损失/频繁止损, 低=能扛回撤")

# 预定义 7 个覆盖极端和中间状态的人格原型：
PRESET_PROFILES: dict[str, OceanProfile] = {
    "冷静创新型":   OceanProfile(name="冷静创新型",   openness=90, conscientiousness=80, extraversion=25, agreeableness=20, neuroticism=10),
    "保守焦虑型":   OceanProfile(name="保守焦虑型",   openness=15, conscientiousness=85, extraversion=20, agreeableness=70, neuroticism=90),
    "激进冒险型":   OceanProfile(name="激进冒险型",   openness=85, conscientiousness=20, extraversion=80, agreeableness=15, neuroticism=10),
    "纪律动量型":   OceanProfile(name="纪律动量型",   openness=50, conscientiousness=90, extraversion=75, agreeableness=50, neuroticism=30),
    "逆向价值型":   OceanProfile(name="逆向价值型",   openness=60, conscientiousness=75, extraversion=10, agreeableness=10, neuroticism=25),
    "平衡中庸型":   OceanProfile(name="平衡中庸型",   openness=50, conscientiousness=50, extraversion=50, agreeableness=50, neuroticism=50),
    "情绪追涨型":   OceanProfile(name="情绪追涨型",   openness=70, conscientiousness=15, extraversion=90, agreeableness=80, neuroticism=75),
}
```

### 2. `personality/trait_to_constraint.py`

```python
"""
将OCEAN人格分数映射为硬编码的交易约束参数。
这些约束是不可被LLM推理覆盖的"刚性规则"。

映射逻辑:
- Neuroticism → 止损百分比、最大回撤容忍度、仓位上限
- Conscientiousness → 是否强制止损、是否要求交易日志、再平衡频率
- Openness → 允许交易的币种范围、最大同时持仓数
- Extraversion → 是否使用情绪数据、动量追踪权重
- Agreeableness → 是否参考市场共识信号、从众系数
"""
from pydantic import BaseModel

class TradingConstraints(BaseModel):
    max_position_pct: float        # 单笔最大仓位占总资金百分比
    stop_loss_pct: float           # 止损百分比（触发即强制平仓）
    max_drawdown_pct: float        # 最大回撤容忍度
    max_concurrent_positions: int  # 最大同时持仓数
    rebalance_interval_seconds: int  # 再平衡检查间隔
    allowed_assets: list[str]      # 允许交易的资产列表（"*"表示全部）
    use_sentiment_data: bool       # 是否使用情绪数据
    momentum_weight: float         # 动量信号权重 (0.0 - 1.0)
    contrarian_weight: float       # 逆向信号权重 (0.0 - 1.0)
    require_stop_loss: bool        # 是否强制要求每笔交易设止损
    min_confidence_threshold: float  # 最低信心阈值才允许开仓 (0.0 - 1.0)

def ocean_to_constraints(profile: "OceanProfile", global_config: dict) -> TradingConstraints:
    """
    核心映射函数。所有数值必须有明确的线性或分段映射公式，不允许魔法数字。
    
    映射规则（必须严格实现）:
    
    max_position_pct = clip(5 + (100 - N) * 0.25, 5, 30)
        → N=100时仓位5%, N=0时仓位30%
    
    stop_loss_pct = clip(1 + (100 - N) * 0.14, 1, 15)
        → N=100时止损1%, N=0时止损15%
    
    max_drawdown_pct = clip(2 + (100 - N) * 0.18, 2, 20)
        → N=100时最大回撤2%, N=0时20%
    
    max_concurrent_positions = clip(1 + O // 20, 1, 6)
        → O=100时6个, O=0时1个
    
    rebalance_interval_seconds:
        N > 70 → 300 (5分钟)
        N > 40 → 3600 (1小时)
        N <= 40 → 86400 (1天)
    
    allowed_assets:
        O > 60 → global_config["all_assets"] (全部)
        O <= 60 → global_config["major_assets"] (仅BTC/ETH等主流)
    
    use_sentiment_data = E > 50
    momentum_weight = E / 100.0
    contrarian_weight = (100 - A) / 100.0
    require_stop_loss = C > 50
    min_confidence_threshold = clip(C / 100.0 * 0.8, 0.2, 0.8)
    """
    # 请严格按上述公式实现，不要遗漏任何字段
    ...
```

### 3. `personality/prompt_generator.py`

```python
"""
将OceanProfile转化为LLM的System Prompt。

Prompt结构（必须严格遵循）:

1. 角色设定段 — 你是一个加密货币交易员，具有以下性格特征...
2. 五维人格描述段 — 逐维度描述分数及其交易行为含义
3. 硬约束注入段 — 明确列出TradingConstraints中的所有数值限制
4. 决策输出格式段 — 强制要求JSON输出格式
5. 禁止事项段 — 禁止编造数据、禁止超出约束、禁止输出非JSON

生成的prompt必须:
- 全英文（LLM英文推理更准确）
- 包含所有5个维度的具体分数和行为描述
- 包含所有TradingConstraints的数值（作为HARD CONSTRAINTS段落）
- 以 "You MUST respond with ONLY a valid JSON object" 结尾
"""

def generate_system_prompt(profile: "OceanProfile", constraints: "TradingConstraints") -> str:
    """生成完整的System Prompt字符串"""
    ...

def generate_decision_prompt(
    market_data: dict,    # 当前行情数据
    positions: list,      # 当前持仓
    memory_context: str,  # 记忆模块提供的上下文
    portfolio_value: float  # 当前总资产
) -> str:
    """
    生成每次决策时的User Prompt。
    
    必须包含:
    - 当前价格、24h涨跌幅、成交量
    - 当前持仓明细
    - 记忆上下文（最近N次交易的结果和反思）
    - 当前总资产和可用余额
    
    必须要求LLM输出以下JSON格式:
    {
        "action": "BUY" | "SELL" | "HOLD",
        "asset": "BTC-PERP",
        "size_pct": 15.0,          // 占总资产百分比
        "entry_price": 67200.0,
        "stop_loss_price": 64000.0,
        "take_profit_price": 72000.0,
        "confidence": 0.75,         // 0.0-1.0
        "reasoning": "...",         // 决策理由（1-3句话）
        "personality_influence": "..." // 哪个性格维度主导了这个决策
    }
    """
    ...
```

### 4. `agent/memory.py`

```python
"""
三层记忆系统，模仿FinMem的设计。

Layer 1 — 短期记忆 (Working Memory)
    - 最近 20 条tick级别的价格变动
    - 最近 5 次交易的即时结果
    - 过期策略: FIFO，超过20条自动淘汰最旧的
    - 存储在内存中（Python list）

Layer 2 — 中期记忆 (Episodic Memory)  
    - 每次交易的完整记录（开仓/平仓价格、盈亏、持仓时间、当时的reasoning）
    - 最近 50 笔交易
    - 过期策略: 保留最近50笔，超出按时间淘汰
    - 存储在Redis中（JSON序列化）

Layer 3 — 长期记忆 (Semantic Memory)
    - Agent的交易反思总结（由reflection模块生成）
    - 格式: "我在高波动市场中倾向于过早止损，应该更耐心" 这样的自然语言总结
    - 每完成 10 笔交易触发一次反思
    - 最多保留 20 条反思
    - 存储在Redis中

提供 `get_context_for_decision()` 方法:
    - 从三层记忆中提取与当前决策相关的上下文
    - 短期: 全部输出
    - 中期: 最近5笔交易的摘要
    - 长期: 全部反思总结
    - 返回一个格式化的字符串，直接插入到decision prompt中
"""
```

### 5. `agent/trading_agent.py`

```python
"""
核心交易Agent实现。

每个Agent是一个独立的asyncio Task，生命周期:

1. __init__(profile, constraints, llm_config, market_feed, redis_bus)
2. start() → 启动主循环
3. 主循环:
   a. 等待 rebalance_interval_seconds
   b. 从 market_feed 获取最新行情
   c. 从 memory 获取决策上下文
   d. 构造 decision prompt
   e. 调用 LLM (通过 litellm) 获取决策JSON
   f. 解析JSON，校验是否符合constraints（不符合则拒绝执行）
   g. 如果confidence >= min_confidence_threshold 且通过约束校验:
      - 生成 TradeSignal
      - 推送到 Redis channel "agent_signals"
      - 更新 memory (Layer 1 + Layer 2)
   h. 每10笔交易触发一次 reflection
4. stop() → 优雅关闭

关键设计原则:
- LLM只负责"建议"，硬约束由代码强制执行
- LLM输出的任何超出constraints的值都会被clip到合法范围
- 如果LLM返回无法解析的JSON，跳过本轮决策，记录错误日志
- 每个Agent独立运行，互不干扰
"""

class TradingAgent:
    def __init__(self, 
                 agent_id: str,
                 profile: OceanProfile,
                 constraints: TradingConstraints,
                 llm_config: dict,
                 market_feed: "DataFeed",
                 redis_bus: "RedisBus",
                 telegram: "TelegramNotifier"):
        ...
    
    async def start(self):
        """启动Agent主循环"""
        ...
    
    async def _decision_cycle(self):
        """单次决策循环"""
        ...
    
    def _validate_signal(self, raw_signal: dict) -> TradeSignal | None:
        """
        校验LLM输出是否符合constraints。
        不符合的字段clip到合法范围，严重违规（如买入非允许资产）则返回None。
        """
        ...
    
    async def stop(self):
        """优雅关闭"""
        ...
```

### 6. `execution/signal.py`

```python
"""交易信号数据结构"""
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeSignal(BaseModel):
    agent_id: str
    agent_name: str               # 人格原型名称
    timestamp: datetime
    action: Action
    asset: str
    size_pct: float               # 占该Agent分配资金的百分比
    entry_price: float
    stop_loss_price: float | None
    take_profit_price: float | None
    confidence: float             # 0.0 - 1.0
    reasoning: str
    personality_influence: str    # 哪个OCEAN维度主导
    ocean_profile: dict           # 完整OCEAN分数快照
```

### 7. `execution/aggregator.py`

```python
"""
多Agent信号聚合器。

支持两种聚合模式（通过配置切换）:

模式A — 独立模式 (independent):
    - 每个Agent的信号直接转发给对应的paper_trader
    - 各Agent独立计算PnL
    - 用于对比实验

模式B — 投票模式 (voting):
    - 收集同一时间窗口内所有Agent的信号
    - 按 confidence 加权投票
    - 权重 = confidence × agent_historical_sharpe（历史表现好的Agent权重更高）
    - 多数派决定最终方向
    - 仓位 = 各Agent建议仓位的加权平均
    - 用于实盘

聚合器订阅 Redis channel "agent_signals"，
聚合后的最终信号发布到 Redis channel "aggregated_signals"。
"""
```

### 8. `execution/paper_trader.py`

```python
"""
纸上交易执行器。

功能:
- 维护每个Agent的虚拟账户（初始资金可配置，默认$10,000）
- 接收TradeSignal并模拟执行（按信号价格成交，不考虑滑点）
- 实时跟踪每个Agent的:
  - 当前持仓明细
  - 已实现PnL
  - 未实现PnL
  - 总资产价值
  - Sharpe Ratio（滚动计算）
  - 最大回撤
  - 胜率
  - 盈亏比
- 检查止损/止盈是否触发（每次行情更新时检查）
- 所有交易记录持久化到Redis

提供 `get_agent_stats(agent_id) -> dict` 接口供dashboard调用。
提供 `get_leaderboard() -> list[dict]` 接口返回所有Agent按Sharpe排序的排行榜。
"""
```

### 9. `agent/reflection.py`

```python
"""
Agent交易后反思模块。

每完成10笔交易后触发。
将最近10笔交易的完整记录（含reasoning和结果）发送给LLM，
要求其以Agent的人格角度进行反思。

反思prompt要求LLM输出:
{
    "lessons": ["..."],           // 3条关键教训
    "personality_observation": "...",  // 人格对交易的影响观察
    "adjustment_suggestion": "...",    // 建议（但不会自动执行人格参数变更）
    "emotional_state": "...",         // 当前"情绪状态"描述
    "summary": "..."                  // 一句话总结，存入长期记忆
}

反思结果:
1. summary 存入 Memory Layer 3
2. 完整反思记录通过 Telegram 推送
3. adjustment_suggestion 仅作为参考日志，不自动修改OCEAN参数
   （未来Phase可实现自动调参，但Phase 1不做）
"""
```

### 10. `market/data_feed.py`

```python
"""
行情数据源抽象层。

Phase 1 实现两个数据源:

1. MockDataFeed — 用于测试
   - 从CSV文件加载历史K线数据
   - 按时间顺序逐条推送
   - 支持加速回放（1秒 = 1小时/1天）

2. LiveDataFeed — 用于实盘纸上交易
   - 通过REST API定时拉取行情（避免WebSocket复杂度）
   - 支持的数据源: CoinGecko免费API 或 Binance公开API
   - 拉取间隔: 可配置（默认60秒）
   - 提供统一的数据结构:

class MarketSnapshot(BaseModel):
    timestamp: datetime
    asset: str            # e.g. "BTC-PERP"
    price: float
    price_24h_change_pct: float
    volume_24h: float
    high_24h: float
    low_24h: float

所有DataFeed实现统一的抽象接口:
    async def subscribe(self, assets: list[str]) -> AsyncIterator[MarketSnapshot]
    async def get_latest(self, asset: str) -> MarketSnapshot
"""
```

### 11. `integration/telegram_notifier.py`

```python
"""
Telegram通知推送。

推送以下事件:
1. Agent开仓/平仓信号（含完整reasoning）
2. 止损/止盈触发
3. Agent反思报告（每10笔交易后）
4. 每日汇总报告（各Agent PnL排行榜）
5. 系统异常告警

消息格式（Markdown）:

开仓信号:
🧠 <Agent名称> (O{}/C{}/E{}/A{}/N{})
📊 {action} {asset} @ ${price}
💰 Size: {size_pct}% | SL: ${sl} | TP: ${tp}
🎯 Confidence: {confidence}
💭 {reasoning}
🔑 主导维度: {personality_influence}

日报:
📈 Daily Report - {date}
| Rank | Agent | PnL | Sharpe | MaxDD | Trades |
|------|-------|-----|--------|-------|--------|
| 1    | ...   | ... | ...    | ...   | ...    |

从环境变量读取:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
"""
```

### 12. `scripts/dashboard.py`

```python
"""
Rich终端实时仪表盘。

使用 rich.live 实现自动刷新的终端UI，显示:

1. 顶部: 系统状态（运行时长、活跃Agent数、总信号数）
2. 中部: Agent状态表格
   | Agent | OCEAN | 持仓 | PnL | Sharpe | MaxDD | 最新信号 | 信心度 |
3. 底部: 最近5条交易信号的实时滚动日志

刷新频率: 每2秒从Redis拉取最新数据。
"""
```

---

## 配置文件规范

### `config/agents.yaml`

```yaml
# 使用预定义原型或自定义OCEAN参数
agents:
  - id: "agent_calm_innovator"
    preset: "冷静创新型"          # 使用预定义原型
    initial_capital: 10000

  - id: "agent_conservative"
    preset: "保守焦虑型"
    initial_capital: 10000

  - id: "agent_aggressive"
    preset: "激进冒险型"
    initial_capital: 10000

  - id: "agent_custom_1"
    custom:                       # 自定义OCEAN参数
      name: "自定义测试型"
      openness: 70
      conscientiousness: 60
      extraversion: 40
      agreeableness: 30
      neuroticism: 55
    initial_capital: 10000
```

### `config/trading.yaml`

```yaml
trading:
  assets:
    major: ["BTC-PERP", "ETH-PERP"]
    all: ["BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "DOGE-PERP"]
  
  data_feed:
    type: "live"                 # "mock" or "live"
    source: "binance"            # "binance" or "coingecko"
    interval_seconds: 60
    mock_csv_path: "data/btc_1h_2024.csv"  # mock模式用
  
  aggregator:
    mode: "independent"          # "independent" or "voting"
    signal_window_seconds: 120   # voting模式下信号收集窗口
  
  risk:
    global_max_drawdown_pct: 25  # 全局最大回撤，触发全部Agent暂停
    global_max_daily_loss_pct: 10
```

### `config/llm.yaml`

```yaml
llm:
  provider: "anthropic"          # litellm支持的provider
  model: "claude-sonnet-4-20250514"
  temperature: 0.3               # 低温度保证决策一致性
  max_tokens: 1024
  timeout_seconds: 30
  retry_count: 3
  retry_delay_seconds: 5
  
  # 成本控制
  max_calls_per_agent_per_hour: 12  # 每Agent每小时最多调用12次
  fallback_model: "gpt-4o-mini"     # 主模型失败时的fallback
```

---

## 实现顺序（严格按此顺序，每完成一步确认测试通过后再进下一步）

### Step 1: 基础设施层
1. `utils/config_loader.py` — YAML配置加载
2. `utils/logger.py` — loguru配置
3. 所有配置文件 — `config/*.yaml`
4. `pyproject.toml` — 依赖声明

### Step 2: 人格引擎
5. `personality/ocean_model.py` — OceanProfile + 预定义原型
6. `personality/trait_to_constraint.py` — OCEAN → TradingConstraints映射
7. `personality/prompt_generator.py` — System Prompt + Decision Prompt生成
8. **测试**: `tests/test_personality/` — 验证所有映射公式正确

### Step 3: 数据层
9. `market/data_feed.py` — MockDataFeed + LiveDataFeed
10. `market/indicators.py` — 基础技术指标（RSI、SMA、MACD，用于丰富decision prompt）
11. `execution/signal.py` — TradeSignal数据结构
12. `integration/redis_bus.py` — Redis pub/sub封装

### Step 4: Agent核心
13. `agent/memory.py` — 三层记忆系统
14. `agent/base_agent.py` — Agent基类
15. `agent/trading_agent.py` — 完整交易Agent
16. **测试**: `tests/test_agent/` — 用MockDataFeed + 固定LLM响应测试完整决策流程

### Step 5: 执行层
17. `execution/paper_trader.py` — 纸上交易器
18. `execution/aggregator.py` — 信号聚合器
19. `execution/risk_manager.py` — 全局风控
20. **测试**: `tests/test_execution/`

### Step 6: 集成层
21. `integration/telegram_notifier.py` — Telegram推送
22. `agent/reflection.py` — 反思模块
23. `main.py` — 主入口，启动所有组件

### Step 7: 工具
24. `scripts/dashboard.py` — Rich终端仪表盘
25. `scripts/create_agents_config.py` — 批量配置生成（如生成20个随机OCEAN组合）
26. `scripts/backtest.py` — 历史回测

---

## 编码规范

1. **所有函数必须有Google-style docstring**，包含Args、Returns、Raises
2. **所有Pydantic model必须有Field description**
3. **类型注解覆盖率100%** — 所有函数参数和返回值都必须有类型注解
4. **异步优先** — 所有IO操作使用async/await
5. **错误处理**:
   - LLM调用失败 → 重试3次，最终跳过本轮
   - JSON解析失败 → 记录原始响应到日志，跳过本轮
   - Redis连接断开 → 自动重连，期间Agent暂停决策
   - 行情数据异常 → 跳过本轮，不做决策
6. **不允许在任何模块中硬编码API密钥** — 一律从环境变量读取
7. **LLM调用必须通过litellm** — `from litellm import acompletion`，不直接调用任何provider SDK
8. **所有金额计算使用Decimal** — `from decimal import Decimal`，避免浮点精度问题

---

## 环境变量

```bash
# 必须
LITELLM_API_KEY=sk-...           # 或 ANTHROPIC_API_KEY / OPENAI_API_KEY
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# 可选
BINANCE_BASE_URL=https://api.binance.com  # 行情数据源
LOG_LEVEL=INFO
```

---

## 验收标准

项目完成后必须满足以下条件:

1. `pytest tests/ -v` 全部通过
2. 能用 `python -m src.main` 启动系统，3个预定义Agent并行运行
3. 使用MockDataFeed能完成完整的回测流程，生成各Agent的PnL对比报告
4. Telegram正确推送交易信号和日报
5. Rich仪表盘正确显示所有Agent的实时状态
6. 不同OCEAN参数的Agent确实表现出不同的交易行为:
   - 高N Agent的交易频率和止损频率明显高于低N Agent
   - 高O Agent的交易币种范围大于低O Agent
   - 高E Agent更倾向于追涨（与市场方向一致的交易比例更高）
7. LLM调用成本可控——3个Agent在60秒间隔下，每小时总调用量 ≤ 36次

---

## 禁止事项

1. **禁止**使用MBTI或其他类型论模型 — 只用Big Five连续维度
2. **禁止**让LLM直接控制仓位金额 — LLM只输出百分比建议，实际金额由代码根据constraints计算
3. **禁止**在Agent间共享记忆 — 每个Agent的记忆完全隔离
4. **禁止**自动修改OCEAN参数 — Phase 1中人格参数在运行期间固定不变
5. **禁止**连接真实交易所执行 — Phase 1仅支持paper trading
6. **禁止**使用pandas — 用纯Python + Pydantic处理数据，保持轻量
