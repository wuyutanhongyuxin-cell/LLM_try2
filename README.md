# Personality-Conditioned Multi-Agent Crypto Trading System

> **English | [中文](README_CN.md)**

> A Multi-Agent Crypto Paper Trading System driven by Big Five (OCEAN) Personality Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-253%20passed-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This system uses **Big Five personality theory (OCEAN model)** to create diverse crypto trading agents. Each agent has a unique personality profile that deterministically shapes its trading behavior — risk tolerance, position sizing, asset selection, and decision frequency.

Multiple agents run in parallel, each making independent decisions via LLM (through `litellm`), while hard constraints enforced by code prevent any agent from exceeding its personality-derived limits.

```
                    ┌─────────────────────────────────┐
                    │       OCEAN Personality          │
                    │  O=90 C=80 E=25 A=20 N=10       │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌────────────┐ ┌──────────────┐
      │ System Prompt│ │ Constraints│ │ Memory (3L)  │
      │ (personality)│ │ (hard code)│ │ W / E / S    │
      └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
             │               │               │
             └───────┬───────┘               │
                     ▼                       │
              ┌──────────────┐               │
              │  LLM Call    │◄──────────────┘
              │ (3x voting)  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  Validate &  │  ← clip to constraints
              │  Clip Signal │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Paper Trader │  → PnL - costs (slippage + fees + funding)
              └──────────────┘
```

---

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **LLM suggests, code enforces** | `_validate_signal()` clips all values to constraint ranges |
| **Personality = continuous, not categorical** | OCEAN scores 0-100, not MBTI types |
| **Deterministic constraints** | `trait_to_constraint.py` uses fixed formulas, no LLM influence |
| **Agent isolation** | Each agent has independent memory, positions, and PnL |
| **No float for money** | All financial calculations use `decimal.Decimal` |
| **Realistic costs** | Crypto: slippage + fees + funding rate; CME: slippage + per-contract commission |
| **Multi-sample consistency** | 3 LLM calls per decision with majority voting |
| **Anti look-ahead bias** | Asset anonymization prevents LLM from recalling historical prices |

---

## OCEAN Personality Mapping

Each of the five dimensions maps to specific trading behaviors:

| Dimension | High Score (→100) | Low Score (→0) |
|-----------|-------------------|----------------|
| **O**penness | Trades altcoins, novel strategies | BTC/ETH only, conservative |
| **C**onscientiousness | Strict stop-loss, disciplined | Impulsive, ignores risk mgmt |
| **E**xtraversion | Momentum-chasing, follows crowd | Contrarian, independent |
| **A**greeableness | Herding, aligns with consensus | Challenges consensus, shorts |
| **N**euroticism | Tight stops, frequent cutting | Holds through drawdowns |

### Constraint Formulas (Hard-Coded)

```python
max_position_pct     = clip(5 + (100 - N) * 0.25, 5, 30)
stop_loss_pct        = clip(1 + (100 - N) * 0.14, 1, 15)
max_drawdown_pct     = clip(2 + (100 - N) * 0.18, 2, 20)
max_concurrent_pos   = clip(1 + O // 20, 1, 6)
rebalance_interval   = 300s if N>70, 3600s if N>40, 86400s otherwise
allowed_assets       = all if O>60, major_only otherwise
use_sentiment        = E > 50
momentum_weight      = E / 100
contrarian_weight    = (100 - A) / 100
require_stop_loss    = C > 50
min_confidence       = clip(C * 0.008, 0.2, 0.8)
```

---

## 32 Preset Personality Archetypes (2^5 Binary OCEAN)

Based on the **SLOAN personality classification** — each OCEAN dimension split into High/Low — yielding all 32 unique trading personalities. 4 classic archetypes retain original scores for backward compatibility; 28 new archetypes use H=80, L=20.

### 4 Classic Archetypes (★)

| Archetype | O | C | E | A | N | Code | Trading Style |
|-----------|---|---|---|---|---|------|---------------|
| **Calm Innovator** ★ | 90 | 80 | 25 | 20 | 10 | HHLLL | Explores new assets, disciplined, contrarian |
| **Conservative Anxious** ★ | 15 | 85 | 20 | 70 | 90 | LHLHH | Major assets only, very tight stops, checks every 5min |
| **Aggressive Risk-Taker** ★ | 85 | 20 | 80 | 15 | 10 | HLHLL | All assets, momentum-chasing, loose risk mgmt |
| **Emotional Chaser** ★ | 70 | 15 | 90 | 80 | 75 | HLHHH | FOMO-driven, herding, tight stops |

### O↓C↓ Conservative-Impulsive (8 types)

| Archetype | O | C | E | A | N | Code | Trading Style |
|-----------|---|---|---|---|---|------|---------------|
| Lazy Headwind | 20 | 20 | 20 | 20 | 20 | LLLLL | No discipline, no direction, fully passive |
| Anxious Rebel | 20 | 20 | 20 | 20 | 80 | LLLLH | Fearful contrarian, no risk control |
| Casual Observer | 20 | 20 | 20 | 80 | 20 | LLLHL | Agreeable but passive, waits for consensus |
| Indecisive Worrier | 20 | 20 | 20 | 80 | 80 | LLLHH | Paralyzed by fear and consensus-seeking |
| Gambler Charger | 20 | 20 | 80 | 20 | 20 | LLHLL | Reckless momentum, no stops, calm |
| Nervous Scalper | 20 | 20 | 80 | 20 | 80 | LLHLH | Chases momentum then panics out |
| Retail Follower | 20 | 20 | 80 | 80 | 20 | LLHHL | Follows crowd optimistically, no discipline |
| Panic Follower | 20 | 20 | 80 | 80 | 80 | LLHHH | Herds into trades then panic-sells |

### O↓C↑ Conservative-Disciplined (7 types + 1 classic)

| Archetype | O | C | E | A | N | Code | Trading Style |
|-----------|---|---|---|---|---|------|---------------|
| Iron Defense | 20 | 80 | 20 | 20 | 20 | LHLLL | Fortress mentality, strict rules, contrarian |
| Cautious Sniper | 20 | 80 | 20 | 20 | 80 | LHLLH | Waits patiently, tight stops, few trades |
| Steady Conservative | 20 | 80 | 20 | 80 | 20 | LHLHL | Safe and steady, consensus-aligned |
| Disciplined Striker | 20 | 80 | 80 | 20 | 20 | LHHLL | Follows momentum with strict risk mgmt |
| Calculative Arber | 20 | 80 | 80 | 20 | 80 | LHHLH | Precise entries, tight risk, anxious exit |
| Disciplined Follower | 20 | 80 | 80 | 80 | 20 | LHHHL | Follows trends with discipline and patience |
| Risk-Controlled Trend | 20 | 80 | 80 | 80 | 80 | LHHHH | Full risk control, follows consensus tightly |

### O↑C↓ Exploratory-Impulsive (6 types + 2 classics)

| Archetype | O | C | E | A | N | Code | Trading Style |
|-----------|---|---|---|---|---|------|---------------|
| Wild Hunter | 80 | 20 | 20 | 20 | 20 | HLLLL | Explores exotic assets, no rules, calm |
| Paranoid Innovator | 80 | 20 | 20 | 20 | 80 | HLLLH | Tries new things but panics at drawdowns |
| Zen Explorer | 80 | 20 | 20 | 80 | 20 | HLLHL | Curious but passive, agreeable, relaxed |
| Sensitive Pathfinder | 80 | 20 | 20 | 80 | 80 | HLLHH | Explores cautiously, anxiety-driven exits |
| Restless Speculator | 80 | 20 | 80 | 20 | 80 | HLHLH | High-frequency speculation with panic exits |
| Optimistic Surfer | 80 | 20 | 80 | 80 | 20 | HLHHL | Rides trends on exotic assets, no stops |

### O↑C↑ Exploratory-Disciplined (7 types + 1 classic)

| Archetype | O | C | E | A | N | Code | Trading Style |
|-----------|---|---|---|---|---|------|---------------|
| Precision Contrarian | 80 | 80 | 20 | 20 | 80 | HHLLH | Disciplined counter-trend on diverse assets |
| Calm Researcher | 80 | 80 | 20 | 80 | 20 | HHLHL | Deep analysis, patient, consensus-aware |
| Prudent Observer | 80 | 80 | 20 | 80 | 80 | HHLHH | Careful, disciplined, anxiety-tempered |
| All-Round Dominant | 80 | 80 | 80 | 20 | 20 | HHHLL | Full spectrum: explores, disciplines, leads |
| High-Pressure Elite | 80 | 80 | 80 | 20 | 80 | HHHLH | Peak performance under stress, tight stops |
| Perfect Trend | 80 | 80 | 80 | 80 | 20 | HHHHL | Ideal trend-follower: disciplined + calm |
| All-High Tension | 80 | 80 | 80 | 80 | 80 | HHHHH | All dimensions high, maximum engagement |

---

## Project Structure

```
personality-trading-agents/
├── config/
│   ├── agents.yaml              # Agent personality configs (OCEAN params)
│   ├── trading.yaml             # Trading params, costs, risk, anonymization, debate toggle
│   ├── llm.yaml                 # LLM config + multi-sample + rate limiting
│   └── market_knowledge.json    # Market causal relationship knowledge graph
├── src/
│   ├── personality/             # OCEAN model, constraint mapping, prompt generation (w/ hash), prompt constants
│   ├── agent/                   # Trading agent, multi-sample voting, 3-layer memory, reflection
│   ├── market/                  # Data feeds (Mock/Live/CME Databento), technical indicators, adversarial scenarios
│   ├── execution/               # Signal, paper trader, aggregator, risk mgr, cost model, drift monitor, debate, strategy
│   ├── integration/             # Redis pub/sub, Telegram (signals + drift alerts + cost reports)
│   ├── utils/                   # Config loader, logger, asset anonymizer, trade logger, TF-IDF, knowledge graph
│   └── main.py                  # System entry point
├── tests/                       # 223 tests covering all modules
├── scripts/
│   ├── dashboard.py             # Rich terminal real-time dashboard
│   ├── backtest.py              # Rule-based historical backtesting
│   ├── llm_backtest.py          # Real LLM backtesting with consistency metrics + multi-market
│   ├── generate_synthetic_data.py # Generate synthetic bear/sideways/bull CSV data
│   ├── export_training_data.py  # Export decision traces as JSONL for LLM fine-tuning
│   ├── create_agents_config.py  # Bulk config generation
│   ├── generate_cme_data.py      # Generate synthetic CME futures OHLCV data
│   └── download_cme_data.py      # Download real CME data via Databento API
└── pyproject.toml
```

---

## System Hardening (Phase A-F)

After the initial implementation, the system underwent a comprehensive hardening pass informed by academic research (TradeTrap, Profit Mirage, FINSABER, tau-bench, LiveTradeBench). The following enhancements were added:

### A. Realistic Backtest Engine

**Trading Cost Model** (`cost_model.py`): Every trade incurs real-world costs:

| Cost Component | Default Value | Source |
|----------------|--------------|--------|
| Slippage | 5 bps (0.05%) | Market microstructure |
| Taker fee | 0.04% | Binance perpetual futures |
| Maker fee | 0.02% | Binance perpetual futures |
| Funding rate | 0.015% / 8h | Conservative est., 2024 actual ~0.01-0.017%/8h, BitMEX 78% anchored at 0.01% |

**Asset Anonymization** (`anonymizer.py`): Replaces `BTC-PERP` with `ASSET_A` in prompts to prevent LLM from recalling historical price data. Profit Mirage (2025) showed 51-62% Sharpe decay when removing name-based look-ahead bias.

**LLM Backtest** (`llm_backtest.py`): Real LLM-driven backtest with multi-run consistency:
```bash
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 3 --agents 3 --anonymize
```
Outputs per-agent: avg PnL, PnL std, action agreement rate, pass^k metric.

### B. Agent Decision Stability

**Multi-Sample Voting** (`multi_sample.py`): Each decision calls LLM 3 times (configurable), then majority-votes on the action. Based on Self-Consistency (Wang et al., ICLR 2023): 1→3 samples captures ~80% of consistency gains.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decision_samples` | 3 | LLM calls per decision |
| `consensus_threshold` | 0.6 | Minimum vote share to act (else HOLD) |

**Behavior Drift Detection** (`consistency_monitor.py`): KL divergence between baseline and recent action distributions, with three-tier alerting:

| Severity | KL Threshold | Action |
|----------|-------------|--------|
| Warning | > 0.1 | Log only |
| Critical | > 0.2 | Telegram alert |
| Halt | > 0.5 | Pause agent trading |

**Prompt Versioning**: Every system prompt gets a SHA-256 hash appended (`[prompt_version: abc123...]`), stored in `TradeSignal.prompt_hash` for full traceability.

### C. Memory System Upgrades

**TF-IDF Hybrid Retrieval** (replaces pure rule-based scoring): L2 episodic memory uses a hybrid of TF-IDF semantic similarity (50%) and rule-based scoring (50%) — same asset (+0.5), same action (+0.33), has PnL data (+0.17) — with time decay (0.95^position). Pure Python implementation, no sklearn/numpy.

**Exponential Decay**: L3 semantic memory applies decay weights (alpha=0.98 per position). Recent reflections display in full; older ones show first 50 characters.

### D. Data Layer Fixes

**Accurate 24h Price Change**: MockDataFeed now uses a 24-bar lookback for 24h change calculation (for 1h candles), instead of single-candle open→close which was severely inaccurate.

**Extended MarketSnapshot**: Added `open_price` and `funding_rate` fields for cost model integration.

### E. Per-Agent Risk Management

Global risk manager now includes `check_agent_risk()`: monitors individual agent drawdown and consecutive losses, can halt a single agent without stopping the entire system.

### F. Observability

**Full-Chain Trade Logger** (`trade_logger.py`): Every trade records the complete decision chain — market snapshot, prompt hash, LLM raw response (first 500 chars), pre/post clip signal comparison, clipped fields list, execution result, cost breakdown.

**New Telegram Alerts**: Behavior drift alerts, cost reports alongside existing signal/daily report notifications.

---

## P0: Adversarial Testing (TradeTrap-inspired)

**Adversarial Scenario Generator** (`adversarial.py`): 5 extreme market scenarios based on verified real BTC events:

| Scenario | Real Event | Effect |
|----------|-----------|--------|
| Flash Crash | 2024.3.19 BitMEX: $67K→$8.9K in 2min (spot only) | Single candle -15% |
| Pump | 2024.12.5: BTC breaks $100K | 3 consecutive +5% candles |
| Fake Breakout | 2024 Q1 Grayscale GBTC sell-off | +5% then -9%, net -4% |
| Sideways | 2023 Q3: BTC $25K-$30K range, 50 days | ±1% random per bar |
| V-Reversal | 2024.12: $100K→$93K→$100K | -6% then +6.5% recovery |

**Multi-Market Backtest**: Generate synthetic bear/sideways/bull data and run cross-market comparison:
```bash
python scripts/generate_synthetic_data.py --csv data/btc_1h_2024.csv --output data/
python scripts/llm_backtest.py --csv data/btc_bull.csv --runs 3 --multi-market --anonymize
```

---

## P1: Advanced Memory & Meta-Reflection

**Two-Layer Reflection**: Beyond single-pass reflection (every 10 trades), the system performs **meta-reflection** every 30 trades — analyzing patterns across multiple reflections, identifying strategy evolution and recurring blind spots. Meta-reflections are marked with `[META]` in L3 memory.

**TF-IDF Memory Retrieval** (`tfidf.py`): Pure Python implementation replacing hand-crafted rules. Combines semantic similarity with rule-based scoring:
- TF-IDF cosine similarity on trade reasoning text (50% weight)
- Rule bonuses: same asset (+0.5), same action (+0.33), has PnL (+0.17) (50% weight)
- Time decay: 0.95^position (newer trades weighted higher)

---

## P2: Market Knowledge Graph & Fine-tuning Data Export

**Lightweight Knowledge Graph** (`market_knowledge.json`): A pure-JSON causal relationship map covering BTC/ETH market factors — no Neo4j or external graph DB required.

| Factor | Effect on BTC | Strength | Lag |
|--------|--------------|----------|-----|
| FED_RATE | Negative | Strong | ~30d |
| M2_SUPPLY | Positive | Strong | ~90d |
| DXY | Negative | Moderate | ~7d |
| VIX | Negative | Moderate | 0d |
| BTC_ETF_FLOW | Positive | Strong | ~1d |
| EXCHANGE_RESERVE | Negative (declining=bullish) | Moderate | ~3d |
| FUNDING_RATE | Contrarian | Weak | 0d |
| OPEN_INTEREST | Amplify volatility | Moderate | 0d |

Knowledge context is injected into every Decision Prompt before the memory section, providing agents with macro-level awareness.

**Fine-tuning Data Export** (`export_training_data.py`): Export successful trade decisions as JSONL training data for LLM fine-tuning (LoRA/QLoRA). Uses post-clip signals (validated behavior) as training targets.

```bash
python scripts/export_training_data.py --agent agent_calm_innovator --output data/finetune/
```

---

## P3: Bull/Bear Debate & Execution Strategy Abstraction

**Bull/Bear Debate** (`debate.py`): Inspired by TradingAgents (arxiv 2412.20138), when voting mode has `enable_debate: true`, all agents' reasoning is collected and sent to a neutral judge LLM:

1. Arguments grouped: Bull (BUY) / Bear (SELL) / Neutral (HOLD)
2. Judge outputs: `dominant_view`, `confidence_adjustment` (±0.3), `key_argument`, `risk_flag`
3. BUY signals boosted if BULL dominant, SELL signals boosted if BEAR dominant
4. Only adjusts confidence weights — never changes trade direction

**Important**: This does NOT share agent memories. It only uses the public `reasoning` field from each signal.

**Execution Strategy Abstraction** (`strategy.py`): Decouples signal validation logic from the agent core:

```
ExecutionStrategy (ABC)
  └── RuleBasedStrategy    ← current default (OCEAN constraint clip logic)
  └── RLStrategy           ← future (Phase 2, LLM + RL hybrid)
```

The `_build_signal_from_data()` method now delegates to `strategy.process_signal()`, making it possible to swap in an RL-based execution strategy without modifying agent code.

---

## P4: Multi-Market Support — CME Futures

The system now supports **CME futures** alongside crypto, configurable via `trading.yaml`:

| Market | Assets | Data Source | Costs |
|--------|--------|------------|-------|
| Crypto | BTC-PERP, ETH-PERP, SOL-PERP, ARB-PERP, DOGE-PERP | Binance REST | Slippage + % fees + funding rate |
| CME | ES, NQ, CL, GC, SI, ZB | Databento API | Slippage + per-contract commission |

**CME Contract Specifications** (verified against cmegroup.com):

| Contract | Name | Multiplier | Tick Size | Tick Value |
|----------|------|-----------|-----------|-----------|
| ES | E-mini S&P 500 | $50 | 0.25 | $12.50 |
| NQ | E-mini Nasdaq 100 | $20 | 0.25 | $5.00 |
| CL | Crude Oil | $1,000 | $0.01 | $10.00 |
| GC | Gold | $100 | $0.10 | $10.00 |
| SI | Silver | $5,000 | $0.005 | $25.00 |
| ZB | US Treasury Bond | $1,000 | 1/32 | $31.25 |

**CME Cost Model** — No funding rate, uses per-contract commission:
```yaml
# config/trading.yaml → trading.cme.costs
slippage_bps: 2                # Lower than crypto (2 vs 5 bps)
commission_per_contract: 1.25  # USD per side (broker-only estimate)
enable_costs: true
```

**Prompt Market Awareness**: `prompt_generator.py` adapts role descriptions and trait interpretations per market type (e.g., "explore new altcoins" → "explore diverse futures contracts").

**Data Sources**:
- **Mock**: CSV-based replay via `MockDataFeed` (same as crypto)
- **Live**: `DatabentoCMEFeed` wraps the Databento SDK (requires `DATABENTO_API_KEY`)
- **Synthetic**: `scripts/generate_cme_data.py` creates 2000-bar GBM price paths for all 6 contracts

---

## P5: CME LLM Backtest Fix + Multi-Asset Comparison

Fixed 4 blocking bugs that prevented `llm_backtest.py --market cme` from working:

| Bug | Severity | Fix |
|-----|----------|-----|
| Cost config path broken (`trading.costs` doesn't exist) | Critical | Read from `trading.{market_type}.costs` |
| `account.py` ignores CME cost path | Critical | Added `CMECostConfig` + `contract_multiplier` to `AgentAccount` |
| LLM cost estimate hardcoded to Claude Sonnet ($0.0135/call) | Medium | Auto-detect model: DeepSeek ~$0.001, Claude ~$0.0135, GPT-4o-mini ~$0.0006 |
| No multi-asset comparison mode | Missing | Added `--assets ES CL GC ZB` CLI argument |

**Multi-Asset Comparison** — run backtest across multiple CME contracts and compare:
```bash
python scripts/llm_backtest.py --market cme --assets ES CL GC ZB --runs 2 --anonymize --max-steps 50
```
Outputs per-asset per-agent PnL/Sharpe/trades + cross-asset comparison table.

---

## P6: LLM Backtest Optimization — Zero-Trade Fix

Addressed 5 root causes that caused 2/3 agents (Calm Innovator, Conservative Anxious) to produce **zero trades** in backtests:

| Fix | Root Cause | Severity | Solution |
|-----|-----------|----------|----------|
| **Confidence Scaling** | min_confidence threshold too high for DeepSeek (C=80→0.64, but LLM outputs 0.3-0.6) | Critical | `backtest_confidence_scale: 0.6` in `llm.yaml` — effective threshold = formula × 0.6 |
| **Diagnostic Logging** | Validation failures silently swallowed as "HOLD" | High | `REJECTED` vs `HOLD` distinction + `logger.debug()` on every rejection path |
| **Empty Response Retry** | DeepSeek ~25% empty response rate (known bug) | High | Auto-retry up to 3 times, then mark `EMPTY` |
| **Single-Run Summary** | `--runs 1` skipped `calc_consistency()`, cross-asset table empty | Medium | Always compute consistency (single run → std=0, agreement=1.0) |
| **Open Position Display** | Only closed trades shown, open positions invisible | Low | Trades column: `"0+1open"` format + Actions column with BUY/SELL/HOLD/REJECTED counts |

Initial capital increased to **$5,000,000** per agent for more realistic CME futures sizing.

### P6.1: Root Cause — LLM Outputs HOLD (Not Rejected)

After deploying P6, VPS backtest showed agents still had zero trades. Deeper analysis revealed the **real root cause**: LLM was choosing HOLD itself, not being rejected by confidence threshold. Actions column showed `HOLD:100`, not `REJECTED`.

Three fixes applied:

| Fix | Root Cause | Solution |
|-----|-----------|----------|
| **Logger Import** | `src/utils/logger.py` never imported — `LOG_LEVEL=DEBUG` had no effect | Explicit `import src.utils.logger` in `llm_backtest.py` |
| **System Prompt: Decision Guidelines** | Rules section was all "Do NOT" — LLM defaulted to HOLD as safest choice | Added action-oriented guidance: "You are an ACTIVE trader", confidence calibration (0.4 sufficient), HOLD only when genuinely ambiguous |
| **Decision Prompt: Technical Indicators** | Only price/change/volume — insufficient data for directional conviction | Inject RSI(14) with OVERSOLD/OVERBOUGHT labels, SMA(20) with price relation, MACD histogram with bullish/bearish signal |

Prompt constants extracted to `prompt_constants.py` to keep `prompt_generator.py` under 200 lines.

---

## Three-Layer Memory System (FinMem-inspired)

| Layer | Name | Content | Capacity | Storage | Retrieval |
|-------|------|---------|----------|---------|-----------|
| L1 | Working | Recent 20 ticks + last 5 trade results | 20+5 | In-memory | Full (every decision) |
| L2 | Episodic | Full trade records (price, PnL, reasoning) | 50 trades | Redis | TF-IDF hybrid |
| L3 | Semantic | Reflection summaries (natural language) | 20 entries | Redis | Decay-weighted |

---

## Full Deployment Guide (Ubuntu VPS)

Step-by-step guide for deploying on a fresh Ubuntu 22.04/24.04 VPS.

### Step 1: Server Basics

```bash
ssh root@your-server-ip
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential software-properties-common
```

### Step 2: Install Python 3.11+

```bash
# Ubuntu 22.04 ships Python 3.10 — upgrade via deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version  # should output Python 3.11.x
```

> **Ubuntu 24.04** ships Python 3.12 — skip this step, use `python3` directly.

### Step 3: Install Redis

```bash
sudo apt install -y redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
redis-cli ping  # should output PONG
```

### Step 4: Clone & Install

```bash
cd /opt
git clone https://github.com/wuyutanhongyuxin-cell/LLM_try2.git personality-trading
cd personality-trading
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v  # should see 223 passed
```

### Step 5: Configure Environment

```bash
cp .env.example .env
nano .env
```

```bash
# ── LLM API Key (pick one) ──
DEEPSEEK_API_KEY=sk-...          # cheapest (~$0.001/call)
ANTHROPIC_API_KEY=sk-ant-...     # best quality (~$0.014/call)
OPENAI_API_KEY=sk-...            # mid-range (~$0.001/call)

# ── Infrastructure ──
REDIS_URL=redis://localhost:6379/0

# ── Telegram (optional) ──
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789

# ── CME Data (optional, synthetic data works without this) ──
DATABENTO_API_KEY=db-...

LOG_LEVEL=INFO
```

### Step 6: Choose LLM Model

Edit `config/llm.yaml`:

| `model` value | Provider | Required Key | Cost/call | Best for |
|---------------|----------|-------------|----------|---------|
| `deepseek/deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` | ~$0.001 | Daily testing, bulk backtests |
| `deepseek/deepseek-reasoner` | DeepSeek | `DEEPSEEK_API_KEY` | ~$0.003 | Better reasoning |
| `claude-sonnet-4-20250514` | Anthropic | `ANTHROPIC_API_KEY` | ~$0.014 | Highest decision quality |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` | ~$0.001 | Cost-effective mid-range |

### Step 7: Choose Market Type

Edit `config/trading.yaml`:

```yaml
trading:
  market_type: "crypto"    # change to "cme" for CME futures
```

### Step 8: Generate Synthetic Data

```bash
# Generate CME futures synthetic data (6 contracts × 2000 bars)
python scripts/generate_cme_data.py
# Outputs: data/es_1h_2024.csv, data/cl_1h_2024.csv, etc.
```

### Step 9: Run Your First Backtest

```bash
# Crypto rule-based backtest (free, no LLM calls)
python scripts/backtest.py --market crypto --csv data/btc_1h_2024.csv

# CME rule-based backtest (free)
python scripts/backtest.py --market cme --asset ES --csv data/es_1h_2024.csv
```

### Step 10: Start the Live System

```bash
python -m src.main           # start multi-agent system
# In another terminal:
python scripts/dashboard.py  # real-time Rich dashboard
```

### (Optional) systemd Auto-Start

```bash
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

sudo systemctl daemon-reload
sudo systemctl enable trading-agents
sudo systemctl start trading-agents
journalctl -u trading-agents -f  # view live logs
```

---

## CME Futures Asset Selection Guide

### Supported CME Contracts

| Code | Name | Multiplier | Contract Value (~) | Volatility | Beginner? |
|------|------|-----------|-------------------|-----------|-----------|
| **ES** | E-mini S&P 500 | $50/pt | ~$295,000 | Low | Recommended |
| **NQ** | E-mini Nasdaq 100 | $20/pt | ~$410,000 | Medium | Recommended |
| **CL** | Crude Oil | $1,000/bbl | ~$70,000 | High | Advanced |
| **GC** | Gold | $100/oz | ~$300,000 | Medium-Low | Recommended |
| **SI** | Silver | $5,000/oz | ~$165,000 | High | Advanced |
| **ZB** | US Treasury Bond | $1,000/pt | ~$112,000 | Very Low | Recommended |

### Recommended Combinations

- **Beginner**: `ES GC ZB` — low volatility, clear macro drivers
- **Classic 4-Asset**: `ES CL GC ZB` — equity + commodity + metal + fixed income
- **Full Comparison**: `ES NQ CL GC SI ZB` — all 6 contracts

---

## Backtest Command Reference

### 1. Rule-Based Backtest (Free, No LLM Calls)

```bash
# Crypto
python scripts/backtest.py --market crypto --csv data/btc_1h_2024.csv

# CME — individual contracts
python scripts/backtest.py --market cme --asset ES --csv data/es_1h_2024.csv
python scripts/backtest.py --market cme --asset CL --csv data/cl_1h_2024.csv
python scripts/backtest.py --market cme --asset GC --csv data/gc_1h_2024.csv
python scripts/backtest.py --market cme --asset ZB --csv data/zb_1h_2024.csv
python scripts/backtest.py --market cme --asset NQ --csv data/nq_1h_2024.csv
python scripts/backtest.py --market cme --asset SI --csv data/si_1h_2024.csv
```

### 2. LLM Backtest (Real LLM Calls, Costs API Credits)

```bash
# Quick test (cheap)
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 1 --agents 2 --max-steps 30

# Standard crypto backtest
python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 3 --agents 3 --anonymize

# CME single-asset (e.g., ES, CL, GC, ZB)
python scripts/llm_backtest.py \
  --market cme --asset ES --csv data/es_1h_2024.csv \
  --runs 2 --agents 3 --anonymize --max-steps 100

python scripts/llm_backtest.py \
  --market cme --asset CL --csv data/cl_1h_2024.csv \
  --runs 2 --agents 3 --anonymize --max-steps 100
```

### 3. Multi-Asset Comparison

```bash
# Classic 4-asset
python scripts/llm_backtest.py \
  --market cme --assets ES CL GC ZB \
  --csv-dir data --runs 2 --agents 3 --anonymize --max-steps 50

# All 6 contracts
python scripts/llm_backtest.py \
  --market cme --assets ES NQ CL GC SI ZB \
  --csv-dir data --runs 1 --agents 3 --anonymize --max-steps 30

# Equity indices only
python scripts/llm_backtest.py \
  --market cme --assets ES NQ \
  --csv-dir data --runs 3 --agents 3 --anonymize --max-steps 100

# Commodities only
python scripts/llm_backtest.py \
  --market cme --assets CL GC SI \
  --csv-dir data --runs 2 --agents 3 --anonymize --max-steps 80
```

### 4. Multi-Market Stress Test

```bash
python scripts/generate_synthetic_data.py --csv data/btc_1h_2024.csv --output data/
python scripts/llm_backtest.py --csv data/btc_bull.csv --runs 3 --multi-market --anonymize --max-steps 100
```

### LLM Backtest Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--csv` | string | **required** | Historical data CSV path |
| `--runs` | int | 3 | Number of repeated runs (for consistency metrics) |
| `--agents` | int | 32 | Number of preset personality archetypes to use (32 total) |
| `--anonymize` | flag | off | Enable asset anonymization (BTC-PERP → ASSET_A) |
| `--max-steps` | int | 500 | Max backtest steps (1 step = 1 candle) |
| `--market` | choice | crypto | Market type: `crypto` or `cme` |
| `--asset` | string | auto | Single-asset code (e.g., `ES`, `BTC-PERP`) |
| `--assets` | list | none | Multi-asset mode, space-separated (e.g., `ES CL GC ZB`) |
| `--csv-dir` | string | data | CSV directory for multi-asset mode |
| `--multi-market` | flag | off | Cross-market-regime comparison |

### Cost Estimates

| Scenario | Parameters | DeepSeek | Claude | GPT-4o-mini |
|----------|-----------|----------|--------|-------------|
| Quick test | `--runs 1 --agents 2 --max-steps 30` | ~$0.18 | ~$2.43 | ~$0.11 |
| Standard | `--runs 3 --agents 3 --max-steps 100` | ~$2.70 | ~$36.45 | ~$1.62 |
| 4-asset comparison | `--assets ES CL GC ZB --runs 2 --max-steps 50` | ~$3.60 | ~$48.60 | ~$2.16 |
| Full 6-asset deep | `--assets ES NQ CL GC SI ZB --runs 3 --max-steps 100` | ~$16.20 | ~$218.70 | ~$9.72 |

> Formula: `cost = runs × agents × max_steps × decision_samples(3) × per_call_cost`
> Protected by `max_cost_per_backtest_usd` hard cap (default $50).

---

## Configuration Reference

### Trading Costs (`config/trading.yaml`)

**Crypto** (at `trading.crypto.costs`):
```yaml
trading:
  crypto:
    costs:
      slippage_bps: 5              # 5 bps = 0.05%
      taker_fee_rate: 0.0004       # 0.04%
      maker_fee_rate: 0.0002       # 0.02%
      funding_rate_8h: 0.00015     # 0.015% per 8h
      enable_costs: true
```

**CME** (at `trading.cme.costs`):
```yaml
trading:
  cme:
    costs:
      slippage_bps: 2
      commission_per_contract: 1.25  # USD per side
      enable_costs: true
    contracts:
      ES: { multiplier: 50, tick_size: 0.25, tick_value: 12.50 }
      NQ: { multiplier: 20, tick_size: 0.25, tick_value: 5.00 }
      CL: { multiplier: 1000, tick_size: 0.01, tick_value: 10.00 }
      GC: { multiplier: 100, tick_size: 0.10, tick_value: 10.00 }
      SI: { multiplier: 5000, tick_size: 0.005, tick_value: 25.00 }
      ZB: { multiplier: 1000, tick_size: 0.03125, tick_value: 31.25 }
```

### General Settings
```yaml
trading:
  market_type: "crypto"          # "crypto" | "cme"
  data_feed:
    type: "mock"                 # "mock" (CSV) | "live" (real-time)
    interval_seconds: 60
  aggregator:
    mode: "independent"          # "independent" | "voting"
    enable_debate: false
  anonymize: false               # true recommended for backtests
  risk:
    global_max_drawdown_pct: 25
    global_max_daily_loss_pct: 10
```

### LLM Settings (`config/llm.yaml`)
```yaml
llm:
  model: "deepseek/deepseek-chat"
  temperature: 0.3
  decision_samples: 3            # 1 = single call, 3 = majority voting
  consensus_threshold: 0.6       # 60% vote share to act
  max_calls_per_minute: 20
  max_cost_per_backtest_usd: 50.0
  backtest_confidence_scale: 0.6 # Backtest threshold = formula × 0.6 (fixes zero-trade agents)
```

### Agent Personalities (`config/agents.yaml`)
```yaml
agents:
  - id: "agent_calm_innovator"
    preset: "冷静创新型"           # O=90 C=80 E=25 A=20 N=10
    initial_capital: 10000

  # Custom OCEAN params:
  - id: "agent_custom"
    custom:
      name: "Custom Trader"
      openness: 70
      conscientiousness: 60
      extraversion: 40
      agreeableness: 30
      neuroticism: 55
    initial_capital: 10000
```

---

## Signal Aggregation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `independent` | Each agent's signal executes independently | A/B testing personalities |
| `voting` | Weighted vote: `confidence x historical_sharpe` | Ensemble decisions |
| `voting` + debate | Bull/Bear debate adjusts confidence before voting | Balanced ensemble |

**Bull/Bear Debate** (`debate.py`): When `enable_debate: true`, a neutral judge LLM evaluates all agents' reasoning, grouped into Bull (BUY) / Bear (SELL) / Neutral (HOLD) arguments. The judge returns a `confidence_adjustment` (±0.3 max) that adjusts signal weights without changing trade direction. Inspired by TradingAgents (arxiv 2412.20138).

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.9+ | Async ecosystem |
| LLM Interface | `litellm` | Provider-agnostic (Claude/GPT/local) |
| Data Validation | Pydantic v2 | Type safety + serialization |
| Message Bus | Redis pub/sub | Signal broadcasting |
| Notifications | aiogram 3.x | Telegram alerts + drift warnings |
| Logging | loguru | Structured, colored output |
| Dashboard | rich | Terminal UI |
| CME Data | `databento` | Databento API for CME futures OHLCV |
| Testing | pytest + pytest-asyncio | 253 tests, full coverage |

**Intentionally excluded**: pandas, numpy, django, flask, sqlalchemy (keeping it lightweight).

---

## Telegram Notifications

The system pushes:
- Trade signals with full reasoning
- Stop-loss / take-profit triggers
- Agent reflection reports (every 10 trades)
- Daily leaderboard reports
- **Behavior drift alerts** (KL divergence thresholds)
- **Cost reports** (per-agent accumulated trading costs)

Example signal notification:
```
🧠 Calm Innovator (O90/C80/E25/A20/N10)
📊 BUY BTC-PERP @ $67,200
💰 Size: 25% | SL: $64,000 | TP: $72,000
🎯 Confidence: 0.85
💭 On-chain data shows whale accumulation, RSI oversold
🔑 Dominant: O-dimension — willing to build position during pullback
```

---

## Development Roadmap

### Phase 1 (Complete): Paper Trading Validation
- [x] OCEAN personality model + 32 archetypes (2^5 binary SLOAN coverage)
- [x] Deterministic constraint mapping
- [x] LLM-driven decision loop with hard constraint enforcement
- [x] 3-layer memory system (relevance retrieval + decay)
- [x] Paper trading with full PnL tracking
- [x] Signal aggregation (independent + voting)
- [x] Global + per-agent risk management
- [x] Telegram notifications + drift alerts
- [x] Rich terminal dashboard
- [x] Historical backtesting (rule-based + LLM-driven)

### Phase 1.5 (Complete): System Hardening
- [x] Trading cost model (slippage + fees + funding)
- [x] Multi-sample voting (3x LLM, 60% consensus)
- [x] Asset anonymization (anti look-ahead bias)
- [x] Behavior drift detection (3-tier KL thresholds)
- [x] Prompt version tracking (SHA-256)
- [x] Full-chain trade logging
- [x] Relevance-based memory retrieval
- [x] Exponential memory decay

### P0 (Complete): Adversarial Testing & Multi-Market Backtest
- [x] 5 adversarial scenarios (flash crash, pump, fake breakout, sideways, V-reversal) based on real BTC events
- [x] MockDataFeed adversarial injection support
- [x] Synthetic data generation (bear/sideways/bull markets from single CSV)
- [x] `--multi-market` mode with cross-market comparison table
- [x] Backtest cost cap enforcement (`max_cost_per_backtest_usd`)

### P1 (Complete): Advanced Memory & Meta-Reflection
- [x] Two-layer reflection: L1 reflection (every 10 trades) + L2 meta-reflection (every 30 trades)
- [x] Meta-reflection analyzes patterns across reflections, identifies blind spots
- [x] Pure Python TF-IDF engine (no sklearn/numpy) for semantic memory retrieval
- [x] Hybrid retrieval: TF-IDF similarity (50%) + rule scoring (50%) + time decay

### P2 (Complete): Knowledge Graph & Fine-tuning Data Export
- [x] Lightweight market knowledge graph (`market_knowledge.json`) — BTC/ETH causal relations (macro, on-chain, derivatives)
- [x] Knowledge context injected into Decision Prompt (before memory section)
- [x] Decision trace export script (`export_training_data.py`) — JSONL format for OpenAI/Qwen fine-tuning

### P3 (Complete): Bull/Bear Debate & Execution Strategy Abstraction
- [x] Bull/Bear debate module (`debate.py`) — judge LLM evaluates agent reasoning, adjusts confidence weights
- [x] `enable_debate` toggle in `config/trading.yaml` (default: off)
- [x] ExecutionStrategy interface + RuleBasedStrategy (`strategy.py`) — decouples validation logic from agent core
- [x] Future RL strategies can replace RuleBasedStrategy without modifying agent code

### P4 (Complete): Multi-Market Support — CME Futures
- [x] `config/trading.yaml` multi-market structure (market_type: crypto | cme)
- [x] CME contract specifications (ES/NQ/CL/GC/SI/ZB multipliers, verified against cmegroup.com)
- [x] `databento_feed.py` — CME data source (Live via Databento + Mock fallback)
- [x] `cost_model.py` — CMECostConfig with per-contract commission (no funding rate)
- [x] `prompt_generator.py` market-aware role and trait descriptions
- [x] `market_knowledge.json` extended with CME causal relations

### P5 (Complete): CME LLM Backtest Fix + 4-Asset Comparison
- [x] Fixed cost config path (reads from market-specific section)
- [x] `account.py` + `paper_trader.py` — full CME cost path (commission_per_contract)
- [x] Model-aware LLM cost estimation (DeepSeek $0.001 / Claude $0.0135 / GPT-4o-mini $0.0006)
- [x] `--assets ES CL GC ZB` multi-asset comparison mode with cross-asset table
- [x] `databento_feed.py` graceful ImportError handling

### P6 (Complete): LLM Backtest Optimization
- [x] Configurable confidence scaling (`backtest_confidence_scale: 0.6`) — fixes zero-trade agents
- [x] Diagnostic logging: `REJECTED` vs `HOLD` distinction + debug logs on every rejection
- [x] Empty LLM response auto-retry (up to 3 attempts)
- [x] Single-run consistency calculation (no longer skipped when `--runs 1`)
- [x] Open position display (`"0+1open"` format) + Actions statistics column
- [x] Initial capital increased to $5M for realistic CME futures sizing
- [x] Logger import fix — `LOG_LEVEL=DEBUG` now works for backtest scripts
- [x] System Prompt Decision Guidelines — active trading directives + confidence calibration
- [x] Decision Prompt technical indicators — RSI(14), SMA(20), MACD injected from `indicators.py`
- [x] `prompt_constants.py` extracted from `prompt_generator.py` (file size compliance)

### Phase 2 (Future): Live Trading
- [ ] Connect to real DEX (GRVT/Paradex)
- [ ] Dynamic personality evolution (auto-tuning from reflections)
- [ ] Sentiment data sources (Twitter/Telegram)
- [ ] Voting mode live validation
- [ ] RL strategy replacing RuleBasedStrategy

---

## Academic References

This system's hardening was informed by:
- **Profit Mirage** (2025): LLM trading agents suffer 51-62% Sharpe decay from look-ahead bias
- **Self-Consistency** (Wang et al., ICLR 2023): Multi-sample voting captures ~80% consistency at 3 samples
- **TradeTrap** (2025): Backtest without costs inflates returns by 2-5x
- **tau-bench** (2025): pass@1=61% but pass^8=25% — single-run results are unreliable
- **FinMem** (2023): Three-layer memory with relevance scoring and decay
- **TradingAgents** (2024, arxiv 2412.20138): Bull/Bear researcher debate mechanism
- **TradingGroup** (2025, arxiv 2508.17565): Decision trace collection for LLM fine-tuning
- **Fidelity Digital Assets** (2024): BTC-M2 correlation r=0.78 with ~90d lag

---

## License

MIT
