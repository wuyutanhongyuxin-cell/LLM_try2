# integration — 外部集成

## 用途
Redis 消息总线和 Telegram 通知推送。

## 文件清单
- `redis_bus.py` — Redis pub/sub + KV/List 封装（~121行）
- `telegram_notifier.py` — Telegram 消息推送（~130行）

## 依赖关系
- 本目录依赖：redis, aiogram
- 被以下模块依赖：agent/, execution/, main.py
