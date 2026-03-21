from __future__ import annotations

"""loguru 日志配置 — 自动分类保存到文件。

调用 setup_logging("live") 后，日志同时输出到：
1. 终端（带颜色）
2. logs/live/live_20260320_183000.log（按次运行）
3. logs/live/live_latest.log（最新运行的软链接/覆盖）

支持类型：live / backtest / llm_backtest / main / 自定义
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# 项目根目录（src/utils/logger.py → 上两级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"

# 是否已初始化（防止重复添加 sink）
_initialized = False


def setup_logging(run_type: str = "default", level: str = "") -> Path:
    """初始化日志系统，返回本次运行的日志文件路径。

    Args:
        run_type: 运行类型，决定子目录和文件名前缀
                  "live" → logs/live/live_20260320_183000.log
                  "backtest" → logs/backtest/backtest_20260320_183000.log
                  "llm_backtest" → logs/llm_backtest/llm_backtest_20260320_183000.log
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取

    Returns:
        本次运行的日志文件 Path
    """
    global _initialized  # noqa: PLW0603
    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()

    # 首次调用：重置默认 handler，添加终端 sink
    if not _initialized:
        logger.remove()
        logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
        _initialized = True

    # 创建分类子目录
    type_dir = _LOG_DIR / run_type
    type_dir.mkdir(parents=True, exist_ok=True)

    # 按次运行的日志文件（带时间戳）
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = type_dir / f"{run_type}_{ts}.log"

    # 添加文件 sink（DEBUG 级别全量记录）
    logger.add(
        str(log_file),
        level="DEBUG",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        ),
    )

    # 写一个 latest 指针文件，方便快速找到最新日志
    latest_file = type_dir / f"{run_type}_latest.txt"
    latest_file.write_text(log_file.name, encoding="utf-8")

    logger.info(f"日志已保存到: {log_file}")
    return log_file
