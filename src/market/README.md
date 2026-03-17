# market — 行情数据

## 用途
行情数据源抽象层、技术指标计算和对抗性场景生成。

## 文件清单
- `data_feed.py` — DataFeed 抽象 + MockDataFeed(含对抗注入) + LiveDataFeed（~215行）
- `indicators.py` — RSI / SMA / MACD 技术指标（~78行）
- `adversarial.py` — 对抗性行情场景生成器：5种极端场景（~90行）

## 依赖关系
- 本目录依赖：aiohttp, pydantic
- 被以下模块依赖：agent/, scripts/
