# data/ 目录结构

按市场类型分类存储行情数据和交易记录。

```
data/
├── crypto/                    # 加密货币（BTC, ETH, SOL 等）
│   ├── market/                # 行情 CSV（回测用）
│   └── trades/                # 交易记录 JSONL（回测/纸上交易）
├── cme/                       # CME 期货（ES, CL, GC, ZB, NQ, SI）
│   ├── market/                # 行情 CSV（原油、黄金、国债等）
│   └── trades/                # 交易记录 JSONL
├── lighter/                   # Lighter DEX 实盘
│   └── trades/                # 实盘交易记录 JSONL
├── memory/                    # Agent L4 长期记忆（按 agent_id 分）
│   └── {agent_id}/
│       ├── archive.jsonl      # 反思归档
│       └── wisdom.md          # 投票通过的交易智慧
└── finetune/                  # LLM 微调数据导出
```

## 交易记录格式

每个 JSONL 文件每行一条交易：

```json
{
  "timestamp": "2026-03-20T12:00:00+00:00",
  "agent_id": "live_乐观冲浪型",
  "agent_name": "乐观冲浪型",
  "market_type": "lighter",
  "action": "BUY",
  "asset": "BTC-PERP",
  "size_pct": 20.0,
  "entry_price": 84000.0,
  "stop_loss_price": 83500.0,
  "take_profit_price": 85000.0,
  "confidence": 0.7,
  "reasoning": "RSI oversold...",
  "executed": true,
  "position_after": 0.005,
  "balance_after": 50.0,
  "leverage": 50
}
```
