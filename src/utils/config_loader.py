from __future__ import annotations

"""
YAML 配置加载器。

从 config/ 目录加载所有 YAML 配置文件，
提供统一的配置访问接口。
"""
from pathlib import Path
from typing import Any

import yaml


# 项目根目录下的 config/ 文件夹
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    """加载单个 YAML 配置文件。

    Args:
        filename: 配置文件名，如 "agents.yaml"。

    Returns:
        解析后的字典。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        yaml.YAMLError: YAML 解析失败时抛出。
    """
    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_agents_config() -> dict[str, Any]:
    """加载 Agent 人格配置。"""
    return load_yaml("agents.yaml")


def load_trading_config() -> dict[str, Any]:
    """加载交易参数配置。"""
    return load_yaml("trading.yaml")


def load_llm_config() -> dict[str, Any]:
    """加载 LLM 配置。"""
    return load_yaml("llm.yaml")
