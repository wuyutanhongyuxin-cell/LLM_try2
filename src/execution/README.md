# execution — 执行层

## 用途
交易信号定义、纸上/实盘交易执行、多 Agent 信号聚合、全局风控、辩论机制、执行策略抽象。

## 文件清单
- `signal.py` — TradeSignal 数据结构（~42行）
- `account.py` — AgentAccount 虚拟账户（~178行）
- `paper_trader.py` — PaperTrader 纸上交易管理器（~109行）
- `lighter_executor.py` — Lighter DEX 实盘执行器：IOC 市价单+REST仓位确认（~176行）
- `lighter_helpers.py` — Lighter 辅助函数：下单签名、账户查询（~137行）
- `stats_helper.py` — Sharpe / MaxDD / 胜率 / 盈亏比计算（~75行）
- `aggregator.py` — 信号聚合器 independent + voting + 辩论模式（~110行）
- `risk_manager.py` — 全局风控（~79行）
- `cost_model.py` — 交易成本：滑点+手续费+funding（~90行）
- `consistency_monitor.py` — 行为漂移检测：三级KL阈值（~120行）
- `debate.py` — Bull/Bear 辩论模块（TradingAgents 启发）（~110行）
- `strategy.py` — ExecutionStrategy 接口 + RuleBasedStrategy + confidence兜底（~107行）

## 依赖关系
- 本目录依赖：pydantic, decimal, litellm, lighter-sdk
- 被以下模块依赖：agent/, integration/, main.py, scripts/live_lighter.py
