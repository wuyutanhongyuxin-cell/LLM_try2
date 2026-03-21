# personality — 人格引擎

## 用途
将 Big Five (OCEAN) 人格模型映射为交易约束和 LLM Prompt。

## 文件清单
- `ocean_model.py` — OceanProfile 定义 + 32 个预定义原型（2^5 二元组合，~102行）
- `trait_to_constraint.py` — OCEAN → TradingConstraints 映射公式（~79行）
- `prompt_generator.py` — System Prompt + Decision Prompt 生成，持仓显示含 direction（~205行）

## 依赖关系
- 本目录依赖：pydantic
- 被以下模块依赖：agent/, execution/, main.py
