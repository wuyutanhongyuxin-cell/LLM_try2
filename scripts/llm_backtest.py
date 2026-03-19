from __future__ import annotations

"""真实 LLM 回测：调用 LLM 决策循环，支持多次运行收集一致性数据。

用法: python scripts/llm_backtest.py --csv data/btc_1h_2024.csv --runs 3 --agents 32
特性: 真实 LLM 调用 | --runs N 一致性 | --anonymize 防 bias | 限流控制
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from litellm import acompletion
from loguru import logger
from rich.console import Console

import src.utils.logger as _  # noqa: F401  # 触发 loguru 自定义配置（LOG_LEVEL）

from src.execution.cost_model import CMECostConfig, CostConfig
from src.execution.paper_trader import PaperTrader
from src.market.data_feed import MockDataFeed
from src.personality.ocean_model import PRESET_PROFILES, OceanProfile
from src.personality.prompt_generator import (
    generate_decision_prompt, generate_system_prompt, get_prompt_hash)
from src.personality.trait_to_constraint import ocean_to_constraints
from src.market.indicators import calculate_macd, calculate_rsi, calculate_sma
from src.utils.anonymizer import AssetAnonymizer
from src.utils.config_loader import load_llm_config, load_trading_config

from _backtest_helpers import (  # 同目录辅助模块
    calc_consistency, parse_llm_json, print_results, validate_signal)

console = Console()


def _estimate_per_call_cost(model: str) -> float:
    """根据模型名称估算单次 LLM 调用成本（假设 input ~2000 tokens, output ~500 tokens）。"""
    m = model.lower()
    if "deepseek" in m:
        cost = (2000 * 0.27 + 500 * 1.10) / 1_000_000  # ~$0.001
    elif "claude" in m:
        cost = (2000 * 3 + 500 * 15) / 1_000_000  # ~$0.0135
    elif "gpt-4o-mini" in m:
        cost = (2000 * 0.15 + 500 * 0.60) / 1_000_000  # ~$0.0006
    else:
        cost = (2000 * 3 + 500 * 15) / 1_000_000  # 保守默认
    logger.info(f"LLM 成本估算: ${cost:.6f}/调用 (模型: {model})")
    return cost


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="真实 LLM 回测脚本")
    p.add_argument("--csv", required=True, help="历史数据 CSV 路径")
    p.add_argument("--runs", type=int, default=3, help="重复运行次数（收集一致性）")
    p.add_argument("--agents", type=int, default=32, help="使用前 N 个预定义原型（共32个）")
    p.add_argument("--anonymize", action="store_true", help="启用资产匿名化")
    p.add_argument("--max-steps", type=int, default=500, help="最大回测步数")
    p.add_argument("--market", choices=["crypto", "cme"], default="crypto",
                   help="市场类型 (crypto / cme)")
    p.add_argument("--asset", default="", help="交易资产 (默认: BTC-PERP 或 ES)")
    p.add_argument("--multi-market", action="store_true",
                   help="启用多市况回测（需要 data/ 下有对应的 bear/sideways/bull CSV）")
    p.add_argument("--assets", nargs="+", default=None,
                   help="多品种对比模式，如: --assets ES CL GC ZB")
    p.add_argument("--csv-dir", default="data",
                   help="多品种模式的 CSV 目录（默认 data/，文件名: {asset}_1h_real.csv）")
    return p.parse_args()


def _select_profiles(n: int) -> list[OceanProfile]:
    """选取前 N 个预定义人格原型。"""
    return [PRESET_PROFILES[k] for k in list(PRESET_PROFILES.keys())[:n]]


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


def _build_global_config(trading_cfg: dict, market_type: str = "crypto") -> dict:
    """根据市场类型构建 ocean_to_constraints 所需的全局配置。"""
    return _MARKET_CONFIGS.get(market_type, _MARKET_CONFIGS["crypto"])


async def _run_agent_step(
    agent: dict, snapshot: object, trader: PaperTrader,
    anonymizer: AssetAnonymizer | None, model: str,
    temperature: float, max_tokens: int,
    price_history: list[float] | None = None,
) -> None:
    """单个 Agent 的单步决策：构造 prompt → 调用 LLM → 校验 → 执行。"""
    market_data = {
        "asset": snapshot.asset, "price": snapshot.price,
        "change_24h": snapshot.price_24h_change_pct,
        "volume": snapshot.volume_24h,
    }
    # 注入技术指标，给 LLM 更充分的决策依据
    if price_history and len(price_history) >= 26:
        rsi = calculate_rsi(price_history, 14)
        sma_20 = calculate_sma(price_history, 20)
        macd = calculate_macd(price_history)
        if rsi is not None:
            market_data["rsi_14"] = round(rsi, 2)
        if sma_20 is not None:
            market_data["sma_20"] = round(sma_20, 2)
            market_data["price_vs_sma"] = "above" if snapshot.price > sma_20 else "below"
        if macd is not None:
            market_data["macd_histogram"] = macd["histogram"]
            market_data["macd_signal"] = "bullish" if macd["histogram"] > 0 else "bearish"
    if anonymizer:
        market_data = anonymizer.anonymize_market_data(market_data)
    # 获取持仓信息
    account = trader._accounts[agent["id"]]
    positions_info = [
        {"asset": p.asset, "size": p.size_pct,
         "entry_price": float(p.entry_price), "unrealized_pnl": 0.0}
        for p in account.positions
    ]
    pv = float(account.get_portfolio_value({snapshot.asset: snapshot.price}))
    # Fix 5: 传入 max_positions 让 LLM 了解持仓使用情况
    max_pos = agent["constraints"].max_concurrent_positions
    dec_prompt = generate_decision_prompt(
        market_data, positions_info, "", pv, max_positions=max_pos)
    # 调用 LLM（空响应自动重试，最多 3 次）
    raw = ""
    aid = agent["id"]
    try:
        for attempt in range(3):
            resp = await acompletion(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": agent["sys_prompt"]},
                    {"role": "user", "content": dec_prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
            if raw.strip():
                break
            logger.warning(f"[{aid}] 空响应 (尝试 {attempt + 1}/3)")
    except Exception as e:
        logger.error(f"[{aid}] LLM 调用失败: {e}")
        agent["actions"].append("ERROR")
        return
    if not raw.strip():
        agent["actions"].append("EMPTY")
        return
    parsed = parse_llm_json(raw)
    if parsed is None:
        agent["actions"].append("PARSE_FAIL")
        return
    # Fix 2 + Fix 4: 传入持仓数和持仓状态
    current_positions = len(account.positions)
    has_position = any(p.asset == snapshot.asset for p in account.positions)
    signal = validate_signal(
        parsed, agent["id"], agent["profile"], agent["constraints"],
        snapshot.price, anonymizer, agent["prompt_hash"], model,
        confidence_scale=agent.get("confidence_scale", 1.0),
        current_positions=current_positions,
        has_position=has_position)
    if signal is None:
        # 区分真 HOLD 和被校验拒绝
        llm_action = parsed.get("action", "HOLD").upper()
        if llm_action in ("BUY", "SELL"):
            agent["actions"].append("REJECTED")
        else:
            agent["actions"].append("HOLD")
    else:
        trader.execute_signal(signal)
        agent["actions"].append(signal.action.value)


async def _run_single_backtest(
    profiles: list[OceanProfile], feed_path: str, max_steps: int,
    anonymize: bool, trading_cfg: dict, llm_cfg: dict,
    market_type: str = "crypto", asset: str = "BTC-PERP",
) -> dict[str, dict]:
    """执行单次回测，返回 {agent_id: {name, pnl, sharpe, trades, actions}}。"""
    global_cfg = _build_global_config(trading_cfg, market_type)
    llm = llm_cfg.get("llm", {})
    model = llm.get("model", "claude-sonnet-4-20250514")
    temperature = llm.get("temperature", 0.3)
    max_tokens = llm.get("max_tokens", 1024)
    rpm = llm.get("max_calls_per_minute", 20)
    sleep_between = 60.0 / rpm if rpm > 0 else 3.0
    confidence_scale = llm.get("backtest_confidence_scale", 1.0)
    # 成本硬上限：根据模型自动估算每次调用成本
    # DeepSeek V3: $0.27/M input + $1.10/M output (来源: api-docs.deepseek.com)
    # Claude Sonnet: $3/M input + $15/M output (来源: anthropic.com/pricing)
    # GPT-4o-mini: $0.15/M input + $0.60/M output (来源: openai.com/api/pricing)
    cost_cap = llm.get("max_cost_per_backtest_usd", 50.0)
    per_call_cost = _estimate_per_call_cost(model)
    accumulated_cost = 0.0
    cost_cap_reached = False
    # 根据市场类型读取对应的成本配置
    trading_section = trading_cfg.get("trading", {})
    market_costs = trading_section.get(market_type, {}).get("costs", {})
    if market_type == "cme":
        # CME: 只提取 CostConfig 兼容的字段（佣金由 CMECostConfig 单独处理）
        crypto_compat = {
            "slippage_bps": market_costs.get("slippage_bps", 2),
            "taker_fee_rate": 0.0, "maker_fee_rate": 0.0,
            "funding_rate_8h": 0.0,
            "enable_costs": market_costs.get("enable_costs", True),
        }
        trader = PaperTrader(cost_config=CostConfig(**crypto_compat))
    else:
        trader = PaperTrader(
            cost_config=CostConfig(**market_costs) if market_costs else CostConfig()
        )
    all_assets = global_cfg["all_assets"]
    anonymizer = AssetAnonymizer(all_assets) if anonymize else None
    # 初始化各 Agent
    agents: list[dict] = []
    for i, profile in enumerate(profiles):
        aid = f"agent_{i}"
        constraints = ocean_to_constraints(profile, global_cfg)
        sys_prompt = generate_system_prompt(profile, constraints, market_type)
        if anonymizer:
            sys_prompt = anonymizer.anonymize(sys_prompt)
        if market_type == "cme":
            cme_costs = trading_section.get("cme", {}).get("costs", {})
            cme_cfg = CMECostConfig(
                slippage_bps=cme_costs.get("slippage_bps", 2),
                commission_per_contract=cme_costs.get("commission_per_contract", 1.25),
                enable_costs=cme_costs.get("enable_costs", True),
            )
            contracts_spec = trading_section.get("cme", {}).get("contracts", {})
            multiplier = contracts_spec.get(asset, {}).get("multiplier", 1.0)
            trader.register_agent(
                aid, 5000000.0,
                cme_cost_config=cme_cfg, contract_multiplier=multiplier)
        else:
            trader.register_agent(aid, 5000000.0)
        agents.append({
            "id": aid, "profile": profile, "constraints": constraints,
            "sys_prompt": sys_prompt, "prompt_hash": get_prompt_hash(sys_prompt),
            "actions": [], "confidence_scale": confidence_scale,
        })
    # 步进回测主循环
    feed = MockDataFeed(csv_path=feed_path, asset=asset)
    for step in range(max_steps):
        if cost_cap_reached:
            break
        snapshot = await feed.get_latest(asset)
        if snapshot is None:
            break
        trader.update_prices({asset: snapshot.price})
        # Fix 3: 在 Agent 决策前检查所有持仓的止损/止盈触发
        for agent in agents:
            acct = trader._accounts[agent["id"]]
            events = acct.check_stop_loss_take_profit(asset, snapshot.price)
            for ev in events:
                logger.info(
                    f"[{agent['id']}] {ev['reason']} {ev['asset']} "
                    f"PnL={ev['pnl']:.2f}")
        if step > 0 and step % 24 == 0:  # 每 24 条 K 线记录日收益率
            trader.record_daily_returns()
        for agent in agents:
            interval_steps = max(1, agent["constraints"].rebalance_interval_seconds // 3600)
            if step % interval_steps != 0:
                agent["actions"].append("SKIP")
                continue
            accumulated_cost += per_call_cost
            if accumulated_cost > cost_cap:
                cost_cap_reached = True
                console.print(
                    f"  [bold red][COST CAP REACHED] "
                    f"累计 ${accumulated_cost:.2f} > ${cost_cap:.2f}，停止回测[/bold red]")
                break
            await _run_agent_step(
                agent, snapshot, trader, anonymizer, model, temperature, max_tokens,
                price_history=feed._price_history)
            await asyncio.sleep(sleep_between)  # 限流
        if (step + 1) % 50 == 0:
            console.print(f"  [dim]步骤 {step + 1}/{max_steps}[/dim]")
    # 收集结果
    results: dict[str, dict] = {}
    for agent in agents:
        stats = trader.get_agent_stats(agent["id"])
        account = trader._accounts[agent["id"]]
        results[agent["id"]] = {
            "name": agent["profile"].name,
            "pnl": stats["realized_pnl"] + stats["unrealized_pnl"],
            "sharpe": stats["sharpe_ratio"],
            "trades": stats["total_trades"],
            "open_pos": len(account.positions),
            "actions": agent["actions"],
            "cost_cap_reached": cost_cap_reached,
            "estimated_llm_cost": round(accumulated_cost, 4),
        }
    if cost_cap_reached:
        logger.warning(f"回测在步骤 {step} 因成本上限 ${cost_cap} 提前终止")
    return results


async def _run_multi_market(
    profiles: list[OceanProfile], args: argparse.Namespace,
    trading_cfg: dict, llm_cfg: dict,
) -> None:
    """多市况模式：对 bear/sideways/bull 三种市况分别回测并输出跨市况对比。"""
    from _backtest_helpers import print_cross_market_results
    market_files = {
        "bear": "data/btc_bear.csv",
        "sideways": "data/btc_sideways.csv",
        "bull": "data/btc_bull.csv",
    }
    market_results: dict[str, dict[str, dict]] = {}
    for market_name, csv_path in market_files.items():
        console.print(f"\n[bold magenta]===== 市况: {market_name} ({csv_path}) =====[/bold magenta]")
        all_runs: list[dict[str, dict]] = []
        for run_idx in range(args.runs):
            console.print(f"\n[bold yellow]-- {market_name} Run {run_idx + 1}/{args.runs} --[/bold yellow]")
            result = await _run_single_backtest(
                profiles, csv_path, args.max_steps, args.anonymize, trading_cfg, llm_cfg)
            all_runs.append(result)
        consistency = calc_consistency(all_runs)
        print_results(all_runs, consistency)
        market_results[market_name] = consistency
    # 跨市况对比
    print_cross_market_results(market_results)


async def _run_multi_asset_comparison(
    profiles: list[OceanProfile], args: argparse.Namespace,
    trading_cfg: dict, llm_cfg: dict,
) -> None:
    """多品种对比：对每个资产分别回测，汇总输出对比表。"""
    from _backtest_helpers import print_cross_market_results
    asset_results: dict[str, dict[str, dict]] = {}
    for asset in args.assets:
        csv_path = str(Path(args.csv_dir) / f"{asset.lower()}_1h_real.csv")
        if not Path(csv_path).exists():
            console.print(f"[red]跳过 {asset}: CSV 不存在 ({csv_path})[/red]")
            continue
        console.print(f"\n[bold magenta]===== 品种: {asset} ({csv_path}) =====[/bold magenta]")
        all_runs: list[dict[str, dict]] = []
        for run_idx in range(args.runs):
            console.print(f"\n[bold yellow]-- {asset} Run {run_idx + 1}/{args.runs} --[/bold yellow]")
            result = await _run_single_backtest(
                profiles, csv_path, args.max_steps, args.anonymize,
                trading_cfg, llm_cfg, args.market, asset)
            all_runs.append(result)
        consistency = calc_consistency(all_runs)
        print_results(all_runs, consistency)
        asset_results[asset] = consistency
    if len(asset_results) > 1:
        console.print("\n[bold cyan]===== 跨品种对比 =====[/bold cyan]")
        print_cross_market_results(asset_results)


async def main() -> None:
    """入口：解析参数、执行多轮回测、输出报告。"""
    args = _parse_args()
    trading_cfg = load_trading_config()
    llm_cfg = load_llm_config()
    profiles = _select_profiles(args.agents)
    market = args.market
    asset = args.asset or ("ES" if market == "cme" else "BTC-PERP")
    mkt_label = "CME 期货" if market == "cme" else "加密货币"
    console.print(f"[bold]LLM 回测启动[/bold]: {mkt_label} [{asset}] | "
                  f"{args.runs} 轮 x {len(profiles)} 个 Agent")
    console.print(f"  CSV: {args.csv} | 最大步数: {args.max_steps} | 匿名化: {args.anonymize}")
    if args.assets:
        await _run_multi_asset_comparison(profiles, args, trading_cfg, llm_cfg)
        return
    if args.multi_market:
        await _run_multi_market(profiles, args, trading_cfg, llm_cfg)
        return
    all_runs: list[dict[str, dict]] = []
    for run_idx in range(args.runs):
        console.print(f"\n[bold yellow]-- Run {run_idx + 1}/{args.runs} --[/bold yellow]")
        result = await _run_single_backtest(
            profiles, args.csv, args.max_steps, args.anonymize,
            trading_cfg, llm_cfg, market, asset)
        all_runs.append(result)
    consistency = calc_consistency(all_runs)
    console.print()
    print_results(all_runs, consistency)


if __name__ == "__main__":
    asyncio.run(main())
