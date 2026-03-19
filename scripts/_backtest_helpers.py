from __future__ import annotations

"""LLM 回测辅助函数：信号校验、一致性计算、结果展示。"""

import json
import math
from collections import Counter
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from loguru import logger

from src.execution.signal import Action, TradeSignal
from src.personality.ocean_model import OceanProfile
from src.personality.trait_to_constraint import TradingConstraints
from src.utils.anonymizer import AssetAnonymizer

console = Console()


def parse_llm_json(raw: str) -> dict | None:
    """尝试从 LLM 响应中解析 JSON，失败返回 None。"""
    text = raw.strip()
    # 去除 markdown 代码围栏
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(f"JSON 解析失败，原始响应前100字: {raw[:100]}")
        return None


def _clip(val: float, lo: float, hi: float) -> float:
    """将值裁剪到 [lo, hi] 范围。"""
    return max(lo, min(val, hi))


def validate_signal(
    data: dict, agent_id: str, profile: OceanProfile,
    constraints: TradingConstraints, price: float,
    anonymizer: AssetAnonymizer | None, prompt_hash: str, model: str,
    confidence_scale: float = 1.0,
    current_positions: int = 0,
    has_position: bool = True,
) -> TradeSignal | None:
    """校验 LLM 输出，clip 到合法范围，构造 TradeSignal。

    confidence_scale: 置信度缩放因子，回测时可降低阈值（不修改公式本身）。
    current_positions: 当前持仓数量，用于检查 max_concurrent_positions。
    has_position: 是否有该资产的持仓，SELL 时需要。
    """
    action_str = data.get("action", "HOLD").upper()
    if action_str not in ("BUY", "SELL", "HOLD"):
        logger.debug(f"[{agent_id}] 拒绝: 非法action '{action_str}'")
        return None
    asset = data.get("asset", "BTC-PERP")
    # 反匿名化
    if anonymizer:
        asset = anonymizer.deanonymize_asset(asset)
    # Fix 4: SELL 前检查是否有持仓
    if action_str == "SELL" and not has_position:
        logger.debug(f"[{agent_id}] 拒绝: SELL 但无 {asset} 持仓")
        return None
    # Fix 2: 检查持仓数量是否已满
    if action_str == "BUY" and current_positions >= constraints.max_concurrent_positions:
        logger.debug(
            f"[{agent_id}] 拒绝: 持仓已满 "
            f"{current_positions}/{constraints.max_concurrent_positions}")
        return None
    # 非允许资产的买入 → 拒绝
    if action_str == "BUY" and asset not in constraints.allowed_assets:
        logger.info(f"[{agent_id}] 拒绝买入非允许资产: {asset}")
        return None
    size_pct = _clip(float(data.get("size_pct", 0)), 0, constraints.max_position_pct)
    confidence = _clip(float(data.get("confidence", 0)), 0, 1.0)
    # 代码级兜底：DeepSeek 对非主流品种常输出 conf=0.0 + BUY/SELL 的矛盾组合
    # 此时将 confidence 提升到 0.3（低置信度但可交易），避免全部被拦截
    if action_str in ("BUY", "SELL") and confidence == 0.0:
        confidence = 0.3
        logger.debug(f"[{agent_id}] conf=0.0 兜底 → 0.3 (action={action_str})")
    effective_threshold = constraints.min_confidence_threshold * confidence_scale
    if confidence < effective_threshold:
        logger.debug(
            f"[{agent_id}] 拒绝: conf={confidence:.2f} < "
            f"threshold={effective_threshold:.2f} "
            f"(原={constraints.min_confidence_threshold:.2f}×{confidence_scale})")
        return None
    return TradeSignal(
        agent_id=agent_id, agent_name=profile.name,
        timestamp=datetime.now(tz=timezone.utc),
        action=Action(action_str), asset=asset,
        size_pct=size_pct, entry_price=price,
        stop_loss_price=data.get("stop_loss_price"),
        take_profit_price=data.get("take_profit_price"),
        confidence=confidence,
        reasoning=data.get("reasoning", ""),
        personality_influence=data.get("personality_influence", ""),
        ocean_profile=profile.model_dump(exclude={"name"}),
        prompt_hash=prompt_hash, llm_model=model,
    )


def calc_consistency(all_runs: list[dict[str, dict]]) -> dict[str, dict]:
    """计算多次运行间的一致性指标。"""
    agent_ids = list(all_runs[0].keys())
    report: dict[str, dict] = {}
    for aid in agent_ids:
        pnls = [run[aid]["pnl"] for run in all_runs]
        mean_pnl = sum(pnls) / len(pnls)
        pnl_std = math.sqrt(sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls))
        # action_agreement: 每个时间步多数 action 占比的均值
        action_seqs = [run[aid]["actions"] for run in all_runs]
        min_len = min(len(s) for s in action_seqs)
        agreements: list[float] = []
        for t in range(min_len):
            votes = [s[t] for s in action_seqs]
            if all(v == "SKIP" for v in votes):
                continue
            majority_pct = Counter(votes).most_common(1)[0][1] / len(votes)
            agreements.append(majority_pct)
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0
        # pass^k: PnL > 0 视为"成功"
        success_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
        k = len(pnls)
        report[aid] = {
            "name": all_runs[0][aid]["name"],
            "mean_pnl": mean_pnl, "pnl_std": pnl_std,
            "agreement_rate": agreement_rate,
            "pass_k": success_rate ** k, "k": k,
            "sharpes": [run[aid]["sharpe"] for run in all_runs],
        }
    return report


def print_cross_market_results(
    market_results: dict[str, dict[str, dict]],
) -> None:
    """打印跨市况对比表。market_results = {market_name: consistency_report}"""
    table = Table(title="跨市况对比")
    table.add_column("Agent", style="cyan")
    for market in market_results:
        table.add_column(f"{market} PnL", justify="right")
        table.add_column(f"{market} Sharpe", justify="right")
    # 收集所有 agent_id
    all_aids: set[str] = set()
    for report in market_results.values():
        all_aids.update(report.keys())
    for aid in sorted(all_aids):
        row: list[str] = []
        name = ""
        for market, report in market_results.items():
            data = report.get(aid, {})
            if not name:
                name = data.get("name", aid)
            row.append(f"${data.get('mean_pnl', 0):,.2f}")
            sharpes = data.get("sharpes", [])
            avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
            row.append(f"{avg_sharpe:.3f}")
        table.add_row(name, *row)
    console.print(table)


def print_results(all_runs: list[dict[str, dict]], consistency: dict) -> None:
    """用 Rich 表格打印回测结果。"""
    for i, run in enumerate(all_runs):
        table = Table(title=f"Run {i + 1} 结果")
        table.add_column("Agent", style="cyan")
        table.add_column("PnL", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Actions", justify="left")
        for aid, data in run.items():
            pnl_style = "green" if data["pnl"] > 0 else "red"
            # 交易数：已平仓 + 未平仓标记
            trades_str = str(data["trades"])
            open_pos = data.get("open_pos", 0)
            if open_pos > 0:
                trades_str = f"{data['trades']}+{open_pos}open"
            # 动作统计
            action_counts = Counter(data.get("actions", []))
            action_parts = [
                f"{k}:{v}" for k, v in sorted(action_counts.items())
                if k != "SKIP"
            ]
            action_str = " ".join(action_parts) if action_parts else "-"
            table.add_row(
                data["name"], f"[{pnl_style}]${data['pnl']:,.2f}[/]",
                f"{data['sharpe']:.3f}", trades_str, action_str,
            )
        console.print(table)
    # 一致性报告（有数据就展示）
    if not consistency:
        return
    ct = Table(title="多次运行一致性报告")
    ct.add_column("Agent", style="cyan")
    ct.add_column("Mean PnL", justify="right")
    ct.add_column("PnL Std", justify="right")
    ct.add_column("Agreement", justify="right")
    ct.add_column(f"Pass^{len(all_runs)}", justify="right")
    for aid, data in consistency.items():
        ct.add_row(
            data["name"], f"${data['mean_pnl']:,.2f}",
            f"${data['pnl_std']:,.2f}", f"{data['agreement_rate']:.1%}",
            f"{data['pass_k']:.3f}",
        )
    console.print(ct)
