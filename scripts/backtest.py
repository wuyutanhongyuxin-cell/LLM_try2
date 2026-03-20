from __future__ import annotations

"""历史回测脚本：使用 MockDataFeed + 规则决策（不调LLM）生成 PnL 对比。

支持两种市场:
  加密货币: python scripts/backtest.py --market crypto --csv data/crypto/market/btc_1h_2024.csv
  CME期货:  python scripts/backtest.py --market cme --asset ES --csv data/cme/market/es_1h_2024.csv
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.execution.cost_model import CMECostConfig, CostConfig
from src.execution.paper_trader import PaperTrader
from src.execution.signal import Action, TradeSignal
from src.market.data_feed import MockDataFeed
from src.market.indicators import calculate_rsi
from src.personality.ocean_model import OceanProfile, PRESET_PROFILES
from src.personality.trait_to_constraint import TradingConstraints, ocean_to_constraints
from src.utils.config_loader import load_trading_config

# 市场 → 资产配置
_MARKET_CONFIGS: dict[str, dict[str, list[str]]] = {
    "crypto": {
        "major_assets": ["BTC-PERP", "ETH-PERP"],
        "all_assets": ["BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "DOGE-PERP"],
    },
    "cme": {
        "major_assets": ["ES", "NQ", "CL", "GC", "ZB"],  # 5 个核心 CME 品种
        "all_assets": ["ES", "NQ", "CL", "GC", "SI", "ZB"],  # 含 SI 等扩展品种
    },
}

# 默认 CSV 路径
_DEFAULT_CSV: dict[str, str] = {
    "crypto": "data/crypto/market/btc_1h_2024.csv",
    "cme": "data/cme/market/es_1h_2024.csv",
}

# 默认资产
_DEFAULT_ASSET: dict[str, str] = {
    "crypto": "BTC-PERP",
    "cme": "ES",
}


def _should_buy(
    profile: OceanProfile,
    constraints: TradingConstraints,
    price_change: float,
    rsi: float | None,
) -> bool:
    """根据简化规则判断是否买入。"""
    if profile.conscientiousness > 70:
        if rsi is None or rsi >= 30:
            return False
    if profile.extraversion > 50:
        return price_change > 0
    return price_change < 0


def _make_signal(
    agent_id: str, profile: OceanProfile,
    constraints: TradingConstraints, action: Action,
    price: float, asset: str,
) -> TradeSignal:
    """构造交易信号。止损=stop_loss_pct，止盈=2倍止损。"""
    is_buy = action == Action.BUY
    sl = price * (1 - constraints.stop_loss_pct / 100) if is_buy else None
    tp = price * (1 + constraints.stop_loss_pct * 2 / 100) if is_buy else None
    o, c, e, a, n = (profile.openness, profile.conscientiousness,
                      profile.extraversion, profile.agreeableness, profile.neuroticism)
    return TradeSignal(
        agent_id=agent_id, agent_name=profile.name,
        timestamp=datetime.now(tz=timezone.utc), action=action, asset=asset,
        size_pct=constraints.max_position_pct, entry_price=price,
        stop_loss_price=sl, take_profit_price=tp, confidence=c / 100.0,
        reasoning="backtest_rule",
        personality_influence=f"O={o} C={c} E={e} A={a} N={n}",
        ocean_profile={"O": o, "C": c, "E": e, "A": a, "N": n},
    )


async def run_backtest(csv_path: str, market: str, asset: str) -> list[dict]:
    """执行回测主循环，返回各 Agent 绩效列表。"""
    global_config = _MARKET_CONFIGS[market]
    feed = MockDataFeed(csv_path=csv_path, asset=asset)
    # 根据市场类型加载成本配置
    trading_cfg = load_trading_config()
    trading = trading_cfg.get("trading", {})
    market_costs = trading.get(market, {}).get("costs", {})
    if market == "cme":
        crypto_compat = {
            "slippage_bps": market_costs.get("slippage_bps", 2),
            "taker_fee_rate": 0.0, "maker_fee_rate": 0.0,
            "funding_rate_8h": 0.0,
            "enable_costs": market_costs.get("enable_costs", True),
        }
        trader = PaperTrader(cost_config=CostConfig(**crypto_compat))
    else:
        trader = PaperTrader(
            cost_config=CostConfig(**market_costs) if market_costs else CostConfig())
    # CME 成本详情
    cme_cfg = None
    cme_multiplier = 1.0
    if market == "cme":
        cme_cfg = CMECostConfig(
            slippage_bps=market_costs.get("slippage_bps", 2),
            commission_per_contract=market_costs.get("commission_per_contract", 1.25),
            enable_costs=market_costs.get("enable_costs", True),
        )
        contracts_spec = trading.get("cme", {}).get("contracts", {})
        cme_multiplier = contracts_spec.get(asset, {}).get("multiplier", 1.0)

    agents: list[tuple[str, OceanProfile, TradingConstraints]] = []
    for name, profile in PRESET_PROFILES.items():
        aid = f"bt_{name}"
        cons = ocean_to_constraints(profile, global_config)
        if market == "cme" and cme_cfg is not None:
            trader.register_agent(aid, 100000.0,
                                  cme_cost_config=cme_cfg,
                                  contract_multiplier=cme_multiplier)
        else:
            trader.register_agent(aid, 10000.0)
        agents.append((aid, profile, cons))

    price_history: list[float] = []
    prev_price: float = 0.0
    step: int = 0
    async for snapshot in feed.subscribe([asset]):
        price = snapshot.price
        price_history.append(price)
        change = (price - prev_price) / prev_price * 100 if prev_price > 0 else 0.0
        rsi = calculate_rsi(price_history, period=14)

        trader.update_prices({asset: price})
        for aid, profile, cons in agents:
            if _should_buy(profile, cons, change, rsi):
                sig = _make_signal(aid, profile, cons, Action.BUY, price, asset)
                if sig.confidence >= cons.min_confidence_threshold:
                    trader.execute_signal(sig)

        if step > 0 and step % 24 == 0:
            trader.record_daily_returns()

        prev_price = price
        step += 1
        if step >= 2000:
            break

    return trader.get_leaderboard()


def print_results(results: list[dict], market: str, asset: str) -> None:
    """用 Rich 表格输出回测结果。"""
    console = Console()
    mkt_label = "CME 期货" if market == "cme" else "加密货币"
    table = Table(title=f"回测结果 — {mkt_label} [{asset}] — Agent PnL 对比")
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent", style="cyan")
    table.add_column("总资产", justify="right")
    table.add_column("已实现PnL", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("胜率", justify="right")
    table.add_column("交易数", justify="right")
    table.add_column("总成本", justify="right")

    for i, r in enumerate(results, 1):
        pnl = r["realized_pnl"]
        style = "green" if pnl >= 0 else "red"
        table.add_row(
            str(i), r["agent_id"],
            f"${r['portfolio_value']:,.2f}",
            f"[{style}]${pnl:+,.2f}[/{style}]",
            f"{r['sharpe_ratio']:.2f}",
            f"{r['max_drawdown_pct']:.1f}%",
            f"{r['win_rate']:.1%}",
            str(r["total_trades"]),
            f"${r.get('total_costs', 0):,.2f}",
        )
    console.print(table)


def main() -> None:
    """解析参数并执行回测。"""
    parser = argparse.ArgumentParser(description="Agent 历史回测（支持多市场）")
    parser.add_argument("--market", choices=["crypto", "cme"], default="cme",
                        help="市场类型 (默认: cme)")
    parser.add_argument("--asset", default="", help="交易资产 (默认: 由市场类型决定)")
    parser.add_argument("--csv", default="", help="行情CSV路径 (默认: 自动选择)")
    args = parser.parse_args()

    market = args.market
    asset = args.asset or _DEFAULT_ASSET[market]
    csv_path = args.csv or _DEFAULT_CSV[market]

    console = Console()
    mkt_label = "CME 期货" if market == "cme" else "加密货币"
    console.print(f"[bold]开始回测[/bold]  市场={mkt_label}  资产={asset}  CSV={csv_path}")

    results = asyncio.run(run_backtest(csv_path, market, asset))
    print_results(results, market, asset)


if __name__ == "__main__":
    main()
