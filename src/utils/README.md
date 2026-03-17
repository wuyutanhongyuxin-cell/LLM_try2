# utils — 工具模块

## 用途
配置加载、日志、资产匿名化、交易日志、文本检索和市场知识图谱。

## 文件清单
- `config_loader.py` — YAML 配置文件加载器（~42行）
- `logger.py` — loguru 日志配置（~28行）
- `anonymizer.py` — 资产匿名化防 look-ahead bias（~60行）
- `trade_logger.py` — 全链路交易日志记录器（~58行）
- `tfidf.py` — 纯 Python TF-IDF + Cosine Similarity（~90行）
- `knowledge_graph.py` — 轻量市场知识图谱加载与查询（~80行）

## 依赖关系
- 本目录依赖：pyyaml, loguru, pydantic
- 被以下模块依赖：所有模块
