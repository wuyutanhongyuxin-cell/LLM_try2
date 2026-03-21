#!/usr/bin/env python3
"""Lighter DEX 实盘入口（自动读取链上杠杆）。"""
from __future__ import annotations

import argparse, asyncio, os, sys
from contextlib import suppress
from decimal import Decimal

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logging  # noqa: E402 — 自动日志保存
from src.execution.lighter_executor import LighterExecutor
from src.integration.redis_bus import RedisBus
from src.integration.telegram_notifier import TelegramNotifier
from src.market.lighter_feed import LighterLiveDataFeed
from src.personality.ocean_model import get_profile
from src.personality.trait_to_constraint import ocean_to_constraints
from src.utils.config_loader import load_llm_config, load_trading_config, load_yaml
from src.utils.persistent_trade_logger import PersistentTradeLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lighter DEX 单 Agent 实盘")
    p.add_argument("--agent", default="乐观冲浪型", help="Agent 人格原型名称")
    p.add_argument("--ticker", default="BTC", help="交易对")
    p.add_argument("--capital", type=float, default=100.0, help="标记资本 USD")
    p.add_argument("--interval", type=int, default=0, help="决策间隔秒数（0=用配置）")
    p.add_argument("--max-position", type=float, default=0.0, help="最大仓位 BTC")
    p.add_argument("--dry-run", action="store_true", help="只看信号不下单")
    return p.parse_args()


def _build_asset_config(ticker: str) -> dict[str, list[str]]:
    """构建 ocean_to_constraints 所需的 global_config。"""
    ap = f"{ticker}-PERP"
    try:
        assets = load_trading_config().get("trading", {}).get("crypto", {}).get("assets", {})
        return {"major_assets": assets.get("major", [ap]), "all_assets": assets.get("all", [ap])}
    except Exception:
        return {"major_assets": [ap], "all_assets": [ap]}


async def resolve_market_index(ticker: str) -> int:
    import lighter as sdk
    api = sdk.ApiClient(configuration=sdk.Configuration(host="https://mainnet.zklighter.elliot.ai"))
    obs, _ = await sdk.OrderApi(api).order_books(), await api.close()
    for m in obs.order_books:
        if m.symbol == ticker:
            return m.market_id
    raise RuntimeError(f"Lighter 找不到 ticker: {ticker}")


async def decision_loop(
    feed: LighterLiveDataFeed, executor: LighterExecutor,
    redis_bus: RedisBus, telegram: TelegramNotifier,
    llm_config: dict, profile_name: str, interval: int,
    capital: float, asset_config: dict,
    leverage: int = 1, mmr: float = 0.012,
    trade_logger: PersistentTradeLogger | None = None,
) -> None:
    from src.agent.trading_agent import TradingAgent, _snapshot_to_dict
    from src.personality.prompt_generator import generate_decision_prompt

    profile = get_profile(profile_name)
    constraints = ocean_to_constraints(profile, asset_config)
    agent = TradingAgent(
        agent_id=f"live_{profile_name}", profile=profile,
        constraints=constraints, llm_config=llm_config,
        market_feed=feed, redis_bus=redis_bus,
        leverage=leverage, mmr=mmr,
    )
    agent._portfolio_value = Decimal(str(capital))
    asset = constraints.allowed_assets[0]
    # 从 Redis 恢复交易计数（跨重启延续 L3/L4 触发阈值）
    await agent._restore_trade_count()
    # 预热跳过：第一个 BUY/SELL 信号数据不足，跳过不执行
    warmup_skipped = False

    while True:
        try:
            snapshot = await feed.get_latest(asset)
            if snapshot is None:
                await asyncio.sleep(interval)
                continue
            # 同步真实仓位（多仓/空仓均展示给 LLM）
            real_pos = executor._local_position
            if real_pos != 0:
                direction = "LONG" if real_pos > 0 else "SHORT"
                agent._positions = [{
                    "asset": asset, "size": float(real_pos),
                    "direction": direction,
                    "entry_price": float(executor._last_price or snapshot.price),
                    "unrealized_pnl": 0.0,
                }]
            else:
                agent._positions = []
            bal = await executor.get_balance()
            # #6: 用 balance + margin（notional/leverage）而非 gross notional
            margin = abs(real_pos) * Decimal(str(snapshot.price)) / Decimal(str(leverage))
            agent._portfolio_value = bal + margin
            ctx = await agent._memory.get_context_for_decision(asset, "")
            prompt = generate_decision_prompt(
                _snapshot_to_dict(snapshot), agent._positions,
                ctx, float(agent._portfolio_value),
            )
            n = llm_config.get("decision_samples", 3)
            thr = llm_config.get("consensus_threshold", 0.6)
            if n <= 1:
                raw = await agent._call_llm(prompt)
                sig = agent._validate_signal(raw, snapshot) if raw else None
            else:
                sig = await agent._multi_sample_decision(prompt, snapshot, n, thr)
            if sig is None:
                logger.info(f"[{profile_name}] HOLD")
                await asyncio.sleep(interval)
                continue
            # #14: 预热跳过只在空仓时生效；有持仓时不跳过（避免错过平仓信号）
            if not warmup_skipped and real_pos == 0:
                warmup_skipped = True
                logger.warning(
                    f"[{profile_name}] ⏭️ 预热跳过首信号: "
                    f"{sig.action.value} {sig.asset} @ ${sig.entry_price:,.2f} "
                    f"conf={sig.confidence:.2f} | 原因: 冷启动数据不足"
                )
                await telegram.send_message(
                    f"⏭️ 预热跳过首信号: {sig.action.value} {sig.asset} "
                    f"@ ${sig.entry_price:,.2f} conf={sig.confidence:.2f}"
                )
                await asyncio.sleep(interval)
                continue
            warmup_skipped = True  # 有持仓时直接标记为已预热
            mid = feed.get_mid_price() or Decimal(str(snapshot.price))
            ok = await executor.execute_signal(sig, mid)
            if ok:
                await telegram.notify_signal(sig)
                agent._memory.add_trade_result(sig.model_dump())
                await agent._memory.save_trade_to_l2(sig.model_dump())
                # 磁盘持久化交易记录
                if trade_logger:
                    bal = await executor.get_balance()
                    rec = PersistentTradeLogger.from_signal(
                        sig.model_dump(), agent_id=f"live_{profile_name}",
                        agent_name=profile_name, market_type="lighter",
                        executed=True, position_after=float(executor._local_position),
                        balance_after=float(bal), leverage=leverage,
                    )
                    trade_logger.log_trade(rec)
                agent._trade_count += 1
                await agent._persist_trade_count()
                # 每 10 笔触发 L3 反思（自动归档到 L4）
                if agent._trade_count % 10 == 0:
                    await agent._trigger_reflection()
                # 每 500 笔：10轮LLM投票提取智慧
                if agent._trade_count % 500 == 0:
                    await agent._memory._long_term.extract_wisdom(profile, llm_config)
                # 每 1000 笔：10轮LLM投票淘汰过时经验
                if agent._trade_count % 1000 == 0:
                    recent = await agent._memory.get_recent_trades(10)
                    await agent._memory._long_term.prune_outdated(
                        profile, llm_config, recent,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[{profile_name}] 循环异常: {exc}")
        await asyncio.sleep(interval)


async def main() -> None:
    args = parse_args()
    load_dotenv()
    setup_logging("live")  # 自动保存日志到 logs/live/
    lighter_cfg = load_yaml("lighter.yaml").get("lighter", {})
    llm_cfg = load_llm_config().get("llm", {})
    ticker = args.ticker or lighter_cfg.get("ticker", "BTC")
    interval = args.interval or lighter_cfg.get("default_interval_seconds", 300)
    max_pos = Decimal(str(args.max_position or lighter_cfg.get("max_position_btc", 0.01)))
    leverage = lighter_cfg.get("leverage", 1)
    mmr = lighter_cfg.get("maintenance_margin_rate", 0.004)
    tp_cfg = lighter_cfg.get("tp", {})
    tp_enabled = tp_cfg.get("enabled", False)
    tp_profit_pct = tp_cfg.get("profit_pct", 0.18)
    profile = get_profile(args.agent)
    asset_config = _build_asset_config(ticker)

    market_index = await resolve_market_index(ticker)
    feed = LighterLiveDataFeed(market_index, f"{ticker}-PERP")
    executor = LighterExecutor(
        account_index=int(os.environ.get("LIGHTER_ACCOUNT_INDEX", "0")),
        api_key_index=int(os.environ.get("LIGHTER_API_KEY_INDEX", "0")),
        max_position=max_pos, fill_timeout=lighter_cfg.get("fill_timeout_seconds", 3.0),
        min_balance=Decimal(str(lighter_cfg.get("min_balance_usd", 10))), dry_run=args.dry_run,
        tp_enabled=tp_enabled, tp_profit_pct=tp_profit_pct, leverage=leverage,
    )
    redis_bus = RedisBus(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    telegram = TelegramNotifier()

    try:
        await feed.connect()
        await executor.connect(ticker)
        await redis_bus.connect()
        await telegram.initialize()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    # 链上杠杆优先，配置兜底
    if executor._detected_leverage > 0:
        leverage = executor._detected_leverage
        executor._leverage = leverage
        executor._tp_offset = Decimal(str(tp_profit_pct)) / Decimal(str(leverage))
    logger.info(f"杠杆: {leverage}x {'(链上)' if executor._detected_leverage > 0 else '(配置)'}")

    trade_logger = PersistentTradeLogger(market_type="lighter")
    mode = "DRY-RUN" if args.dry_run else "LIVE"
    o, c, e, a, n = profile.openness, profile.conscientiousness, profile.extraversion, profile.agreeableness, profile.neuroticism
    msg = f"🚀 Lighter [{mode}] {leverage}x | {args.agent} (O{o}/C{c}/E{e}/A{a}/N{n}) | {ticker} {interval}s"
    logger.info(msg)
    await telegram.send_message(msg)

    task = asyncio.create_task(decision_loop(
        feed, executor, redis_bus, telegram,
        llm_cfg, args.agent, interval, args.capital, asset_config,
        leverage=leverage, mmr=mmr, trade_logger=trade_logger,
    ))

    # 用 try/finally 保证 Ctrl+C 时一定执行平仓（兼容所有 Python 版本）
    try:
        await task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("收到退出信号，正在平仓...")
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        try:
            ok = await executor.close_all_positions()
            await telegram.send_message(f"🛑 停止 | 平仓{'成功' if ok else '失败'}")
        except Exception as e:
            logger.error(f"平仓异常: {e}")
        for coro in [feed.disconnect(), executor.disconnect(), redis_bus.close(), telegram.close()]:
            with suppress(Exception):
                await coro
        logger.info("已退出")


if __name__ == "__main__":
    asyncio.run(main())
