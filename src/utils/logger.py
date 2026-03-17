from __future__ import annotations

"""
loguru 日志配置。

统一日志格式，支持通过环境变量 LOG_LEVEL 控制级别。
"""
import os
import sys

from loguru import logger

# 移除默认 handler，重新配置
logger.remove()

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# 控制台输出：带颜色
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)

# 文件输出：按天轮转，保留 7 天
logger.add(
    "logs/{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="7 days",
    encoding="utf-8",
)
