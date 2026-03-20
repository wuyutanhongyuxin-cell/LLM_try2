"""持久化交易记录器 — 逐笔交易写入磁盘 JSONL。

按市场类型分目录存储：
  data/crypto/trades/{agent_id}_trades.jsonl   — 加密货币回测/纸上交易
  data/cme/trades/{agent_id}_trades.jsonl      — CME 期货回测
  data/lighter/trades/{agent_id}_trades.jsonl  — Lighter DEX 实盘
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

# 市场类型 → 目录映射
_MARKET_DIRS: dict[str, str] = {
    "crypto": "data/crypto/trades",
    "cme": "data/cme/trades",
    "lighter": "data/lighter/trades",
}


class TradeRecord(BaseModel):
    """单笔交易的磁盘记录。"""

    timestamp: str = Field(..., description="ISO8601 时间戳")
    agent_id: str = Field(..., description="Agent 标识")
    agent_name: str = Field(default="", description="Agent 人格名称")
    market_type: str = Field(default="crypto", description="市场类型")
    action: str = Field(..., description="BUY / SELL / HOLD")
    asset: str = Field(..., description="交易品种")
    size_pct: float = Field(default=0.0, description="仓位百分比")
    entry_price: float = Field(default=0.0, description="入场价")
    stop_loss_price: float = Field(default=0.0, description="止损价")
    take_profit_price: float = Field(default=0.0, description="止盈价")
    confidence: float = Field(default=0.0, description="信心度")
    reasoning: str = Field(default="", description="LLM 推理")
    executed: bool = Field(default=False, description="是否成功执行")
    position_after: float = Field(default=0.0, description="执行后仓位")
    balance_after: float = Field(default=0.0, description="执行后余额")
    leverage: int = Field(default=1, description="杠杆倍数")


class PersistentTradeLogger:
    """磁盘持久化交易记录器，每笔交易追加写入 JSONL。"""

    def __init__(self, market_type: str = "crypto") -> None:
        dir_str = _MARKET_DIRS.get(market_type, _MARKET_DIRS["crypto"])
        self._dir = Path(dir_str)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._market_type = market_type

    def _path(self, agent_id: str) -> Path:
        """生成 JSONL 文件路径。"""
        safe_id = agent_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_id}_trades.jsonl"

    def log_trade(self, record: TradeRecord) -> None:
        """追加一条交易记录到 JSONL 文件。"""
        path = self._path(record.agent_id)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error(f"交易记录写入失败 [{record.agent_id}]: {exc}")

    def get_trades(self, agent_id: str, last_n: int = 0) -> list[dict]:
        """读取交易记录，last_n=0 返回全部。"""
        path = self._path(agent_id)
        if not path.exists():
            return []
        entries: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if last_n > 0:
            return entries[-last_n:]
        return entries

    def get_trade_count(self, agent_id: str) -> int:
        """获取交易记录总数。"""
        path = self._path(agent_id)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    @staticmethod
    def from_signal(
        signal_dict: dict, agent_id: str, agent_name: str = "",
        market_type: str = "crypto", executed: bool = True,
        position_after: float = 0.0, balance_after: float = 0.0,
        leverage: int = 1,
    ) -> TradeRecord:
        """从 TradeSignal.model_dump() 构建 TradeRecord。"""
        return TradeRecord(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            agent_id=agent_id,
            agent_name=agent_name,
            market_type=market_type,
            action=signal_dict.get("action", "HOLD"),
            asset=signal_dict.get("asset", ""),
            size_pct=signal_dict.get("size_pct", 0.0),
            entry_price=signal_dict.get("entry_price", 0.0),
            stop_loss_price=signal_dict.get("stop_loss_price", 0.0),
            take_profit_price=signal_dict.get("take_profit_price", 0.0),
            confidence=signal_dict.get("confidence", 0.0),
            reasoning=signal_dict.get("reasoning", ""),
            executed=executed,
            position_after=position_after,
            balance_after=balance_after,
            leverage=leverage,
        )
