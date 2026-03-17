from __future__ import annotations

"""Rich 终端实时仪表盘。从 Redis 拉取数据，每2秒刷新。"""

import asyncio
import json
import os
from datetime import datetime, timezone

import redis.asyncio as aioredis
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_start_time: datetime = datetime.now(tz=timezone.utc)
_signal_log: list[str] = []          # 最近信号（保留5条）
_agent_stats: dict[str, dict] = {}   # agent_id -> 统计
_total_signals: int = 0


def _build_header() -> Panel:
    """构建顶部系统状态面板。"""
    elapsed = datetime.now(tz=timezone.utc) - _start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    text = (
        f"运行时长: {hours:02d}:{minutes:02d}:{seconds:02d}  |  "
        f"活跃Agent数: {len(_agent_stats)}  |  "
        f"总信号数: {_total_signals}"
    )
    return Panel(text, title="系统状态", border_style="green")


def _build_agent_table() -> Table:
    """构建中部 Agent 状态表格。"""
    table = Table(title="Agent 绩效", expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent", style="cyan")
    table.add_column("OCEAN", style="magenta")
    table.add_column("持仓", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("最新信号", style="yellow")
    table.add_column("信心度", justify="right")

    # 按 Sharpe 降序排列
    sorted_agents = sorted(
        _agent_stats.values(),
        key=lambda s: s.get("sharpe_ratio", 0),
        reverse=True,
    )
    for i, stat in enumerate(sorted_agents, 1):
        ocean = stat.get("ocean", "N/A")
        pnl_val = stat.get("realized_pnl", 0) + stat.get("unrealized_pnl", 0)
        pnl_style = "green" if pnl_val >= 0 else "red"
        table.add_row(
            str(i),
            stat.get("agent_name", stat.get("agent_id", "?")),
            ocean,
            str(stat.get("open_positions", 0)),
            f"[{pnl_style}]${pnl_val:+.2f}[/{pnl_style}]",
            f"{stat.get('sharpe_ratio', 0):.2f}",
            f"{stat.get('max_drawdown_pct', 0):.1f}%",
            stat.get("last_signal", "-"),
            f"{stat.get('last_confidence', 0):.2f}",
        )
    return table


def _build_signal_log() -> Panel:
    """构建底部信号滚动日志。"""
    if not _signal_log:
        content = "[dim]等待信号...[/dim]"
    else:
        content = "\n".join(_signal_log[-5:])
    return Panel(content, title="最近信号", border_style="blue")


def build_layout() -> Layout:
    """组装完整仪表盘布局。"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=3),
        Layout(name="footer", size=9),
    )
    layout["header"].update(_build_header())
    layout["body"].update(_build_agent_table())
    layout["footer"].update(_build_signal_log())
    return layout


async def _subscribe_signals(rds: aioredis.Redis) -> None:
    """订阅 Redis 信号频道，更新全局状态。"""
    global _total_signals
    pubsub = rds.pubsub()
    await pubsub.subscribe("agent_signals", "agent_stats")
    async for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        channel = msg["channel"]
        if isinstance(channel, bytes):
            channel = channel.decode()
        try:
            data = json.loads(msg["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        if channel == "agent_signals":
            _total_signals += 1
            line = (
                f"[{data.get('timestamp', '?')}] "
                f"{data.get('agent_name', '?')} | "
                f"{data.get('action', '?')} {data.get('asset', '?')} | "
                f"信心={data.get('confidence', 0):.2f}"
            )
            _signal_log.append(line)
        elif channel == "agent_stats":
            aid = data.get("agent_id", "")
            _agent_stats[aid] = data


async def _run_dashboard() -> None:
    """主循环：连接 Redis，启动订阅，每2秒刷新仪表盘。"""
    console = Console()
    rds: aioredis.Redis | None = None
    try:
        rds = aioredis.from_url(REDIS_URL, decode_responses=True)
        await rds.ping()
        console.print("[green]Redis 已连接[/green]")
    except Exception:
        console.print("[yellow]Redis 不可用，等待连接...[/yellow]")
    if rds is not None:
        asyncio.create_task(_subscribe_signals(rds))

    with Live(build_layout(), console=console, refresh_per_second=0.5) as live:
        while True:
            live.update(build_layout())
            await asyncio.sleep(2)


def main() -> None:
    """入口函数。"""
    asyncio.run(_run_dashboard())


if __name__ == "__main__":
    main()
