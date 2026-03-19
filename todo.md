# todo.md — Personality Trading Agents 进度跟踪

## Phase 1：纸上交易验证 ✅

### Step 1-7：核心实现 ✅ (126 tests)
全部完成，详见 git 历史。

### Phase A：回测引擎重建 ✅
- [x] `execution/cost_model.py` — 滑点+手续费+资金费率模型
- [x] 成本模型集成到 `account.py` / `paper_trader.py`
- [x] `scripts/llm_backtest.py` — 真实 LLM 回测（多runs+pass^k）
- [x] `utils/anonymizer.py` — 资产匿名化防 look-ahead bias
- [x] `config/trading.yaml` 新增 costs + anonymize 配置

### Phase B：Agent 稳定性增强 ✅
- [x] `execution/consistency_monitor.py` — 行为漂移检测（三级KL阈值）
- [x] `agent/multi_sample.py` — 多采样投票机制（默认3次）
- [x] `agent/trading_agent.py` 集成多采样+匿名化+prompt hash
- [x] `personality/prompt_generator.py` — Prompt 版本控制(SHA256)
- [x] `execution/signal.py` 新增 prompt_hash + llm_model 字段

### Phase C：记忆系统升级 ✅
- [x] `agent/memory.py` — 相关性检索（替代纯FIFO）
- [x] `agent/memory.py` — 指数衰减机制（L3 alpha=0.98）

### Phase D：数据层增强 ✅
- [x] `market/data_feed.py` — 修复24h变化（用24条前价格）
- [x] `market/data_feed.py` — MarketSnapshot 新增 open_price + funding_rate
- [x] `config/trading.yaml` 成本配置段
- [x] `config/llm.yaml` 多采样+限流配置

### Phase E：全局风控增强 ✅
- [x] `execution/risk_manager.py` — 新增 Agent 级风控

### Phase F：可观测性增强 ✅
- [x] `utils/trade_logger.py` — 全链路交易日志
- [x] `integration/telegram_notifier.py` — 漂移告警+成本报告

### 测试 ✅
- [x] 148 tests 全部通过（原126 + 新增22）
- [x] test_cost_model.py (8 tests)
- [x] test_consistency_monitor.py (7 tests)
- [x] test_anonymizer.py (7 tests)

---

## P0：对抗性测试 + 多市况回测 ✅

### P0-1：对抗性场景注入 ✅
- [x] `market/adversarial.py` — 5种对抗性场景（闪崩/暴涨/假突破/横盘/V反转）
- [x] `market/data_feed.py` — MockDataFeed 支持 adversarial_scenarios 参数注入
- [x] `tests/test_market/test_adversarial.py` — 9 个测试全部通过

### P0-2：多时间窗口回测 ✅
- [x] `scripts/generate_synthetic_data.py` — 从单CSV生成熊市/横盘/牛市合成数据
- [x] `scripts/llm_backtest.py` — 新增 --multi-market 参数，跨市况对比
- [x] `scripts/_backtest_helpers.py` — 新增 print_cross_market_results()
- [x] `scripts/llm_backtest.py` — 成本上限执行逻辑（max_cost_per_backtest_usd）

### Bug Fixes ✅
- [x] `scripts/_backtest_helpers.py` — model_version → llm_model 字段名修复
- [x] `config/trading.yaml` — funding_rate_8h 注释修正

---

## P1：高级记忆 + 元反思 ✅

### P1-1：多层反思机制 ✅
- [x] `agent/reflection.py` — 新增 generate_meta_reflection() 元反思
- [x] `agent/trading_agent.py` — 每30笔交易触发元反思，[META]标记存入L3
- [x] `tests/test_agent/test_trading_agent.py` — 4 个元反思测试

### P1-2：TF-IDF 记忆检索 ✅
- [x] `utils/tfidf.py` — 纯 Python TF-IDF + cosine similarity
- [x] `agent/memory.py` — get_relevant_trades() 改为 TF-IDF 混合检索
- [x] `tests/test_utils/test_tfidf.py` — 10 个 TF-IDF 测试
- [x] `tests/test_agent/test_memory.py` — 3 个 TF-IDF 记忆检索测试

### 测试 ✅
- [x] 174 tests 全部通过（原148 + 新增26）

---

## P2：知识图谱 + 微调数据导出 ✅

### P2-1：轻量知识图谱 ✅
- [x] `config/market_knowledge.json` — BTC/ETH 因果关系图谱（宏观/链上/衍生品）
- [x] `utils/knowledge_graph.py` — 图谱加载与查询（纯 JSON + 标准库）
- [x] `personality/prompt_generator.py` — Decision Prompt 注入 Market Knowledge 段
- [x] `tests/test_utils/test_knowledge_graph.py` — 6 个测试全部通过

### P2-2：决策轨迹导出 ✅
- [x] `scripts/export_training_data.py` — JSONL 微调数据导出（OpenAI/Qwen 格式）

---

## P3：辩论机制 + 执行策略抽象 ✅

### P3-1：Bull/Bear 辩论机制 ✅
- [x] `execution/debate.py` — 裁判 LLM 辩论模块（TradingAgents 启发）
- [x] `execution/aggregator.py` — voting 模式新增 enable_debate 参数
- [x] `config/trading.yaml` — 新增 enable_debate: false
- [x] `tests/test_execution/test_debate.py` — 7 个测试全部通过

### P3-2：执行策略抽象层 ✅
- [x] `execution/strategy.py` — ExecutionStrategy 接口 + RuleBasedStrategy
- [x] `agent/trading_agent.py` — _build_signal_from_data 委托给 Strategy（纯重构）
- [x] `tests/test_execution/test_strategy.py` — 6 个测试全部通过

### 测试 ✅
- [x] 193 tests 全部通过（原174 + 新增19）

---

## P4：多市场支持 — CME 期货 ✅

### P4-1：配置与数据源 ✅
- [x] `config/trading.yaml` — 多市场结构（market_type: crypto | cme），CME 合约规格
- [x] `src/market/databento_feed.py` — Databento CME 数据源（Live + Mock 回退）
- [x] `src/market/data_feed.py` — 新增 CME 默认价格（ES/NQ/CL/GC/SI/ZB）
- [x] `pyproject.toml` — 新增 databento 依赖
- [x] `.env.example` — 新增 DATABENTO_API_KEY

### P4-2：成本模型 ✅
- [x] `src/execution/cost_model.py` — 新增 CMECostConfig + 按手佣金计算（无资金费率）
- [x] `tests/test_execution/test_cost_model.py` — 5 个 CME 成本测试

### P4-3：Prompt 市场感知 ✅
- [x] `src/personality/prompt_generator.py` — 按市场类型切换角色/维度描述/资产示例
- [x] `tests/test_personality/test_prompt_generator.py` — 6 个 CME Prompt 测试

### P4-4：知识图谱扩展 ✅
- [x] `config/market_knowledge.json` — 新增 ES/NQ/CL/GC/SI/ZB 因果关系 + regime 指标

### P4-5：集成与路由 ✅
- [x] `src/main.py` — market_type 路由：数据源/资产/Agent 创建
- [x] `src/agent/trading_agent.py` — 接受 market_type 参数传递给 Prompt
- [x] `tests/conftest.py` — CME_GLOBAL_CONFIG fixture

### 测试 ✅
- [x] 204 tests 全部通过（原193 + 新增11）

---

## P5：CME LLM 回测修复 + 4 品种对比 ✅

### P5-1：成本配置路径修复 ✅
- [x] `scripts/llm_backtest.py` — 修复成本配置读取（`trading.costs` → `trading.{market_type}.costs`）
- [x] `scripts/backtest.py` — 同步修复 CME 成本路径
- [x] LLM 成本估算按模型区分（DeepSeek $0.001 / Claude $0.0135 / GPT-4o-mini $0.0006）

### P5-2：CME 成本路径打通 ✅
- [x] `src/execution/account.py` — 新增 CME 成本路径（CMECostConfig + contract_multiplier）
- [x] `src/execution/paper_trader.py` — register_agent 支持 CME 参数
- [x] `src/market/databento_feed.py` — _fetch_latest_sync 捕获 ImportError 避免崩溃

### P5-3：多品种对比模式 ✅
- [x] `scripts/llm_backtest.py` — 新增 `--assets ES CL GC ZB` 多品种对比参数
- [x] `_run_multi_asset_comparison()` — 分品种回测 + 跨品种对比表

### P5-4：测试 ✅
- [x] `tests/test_market/test_databento_feed.py` — 10 个 Databento/CME 数据源测试
- [x] `tests/test_execution/test_cme_backtest.py` — 9 个 CME 端到端成本路径测试
- [x] 223 tests 全部通过（原 204 + 新增 19）

---

## P8：32→24 Agent 阶段筛选（进行中）

### 阶段 2 初筛 ✅
- [x] 32 Agent × 100 步 × 4 CME 品种（ES/CL/GC/ZB）
- [x] 淘汰 8 个零交易/无差异 Agent
- [x] 保留 24 个：A 级 5 + B 级 8 + C 级 11
- [x] A 级冠军：激进冒险型 +$34,620、情绪追涨型 +$30,339

### 阶段 3 深度回测
- [ ] 第一批（13 个 A+B 级）× 200 步 × 4 品种（~18h）
- [ ] 第二批（11 个 C 级）× 200 步 × 4 品种（~15h）
- [ ] 最终筛选：缩减到 6~8 个 Agent 进长跑验证

---

## Phase 2（未来）
- [ ] 接入真实DEX（GRVT/Paradex）
- [ ] Agent人格动态进化（反思自动调参）
- [ ] 情绪数据源接入（Twitter/Telegram sentiment）
- [ ] 投票模式实盘验证
- [ ] RL 策略替换 RuleBasedStrategy
- [ ] 国内期货市场支持（中金所/上期所/大商所/郑商所 + Amazingdata 数据源）
