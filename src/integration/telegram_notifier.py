"""Telegram 通知推送模块。

使用 aiogram 3.x 发送交易信号、止损告警、日报、反思报告等。
消息格式使用 Markdown，发送失败只记日志不抛异常。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from loguru import logger

try:
    from aiogram import Bot
    from aiogram.enums import ParseMode
except ImportError:
    Bot = None  # type: ignore[assignment,misc]
    ParseMode = None  # type: ignore[assignment,misc]

from src.execution.signal import TradeSignal


class TelegramNotifier:
    """Telegram 消息推送，未启用时所有方法静默返回。"""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._bot: Any = None
        self._enabled = bool(self._token and self._chat_id)

    async def initialize(self) -> None:
        """初始化 Bot 实例。"""
        if self._enabled and Bot is not None:
            self._bot = Bot(token=self._token)
            logger.info("Telegram 通知已启用")
        else:
            logger.warning("Telegram 通知未启用（缺少 token/chat_id 或未安装 aiogram）")

    async def send_message(self, text: str) -> None:
        """发送消息，失败只记日志。"""
        if not self._enabled or self._bot is None:
            return
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN if ParseMode else "Markdown",
            )
        except Exception as exc:
            logger.error("Telegram 发送失败: {}", exc)

    async def notify_signal(self, signal: TradeSignal) -> None:
        """推送交易信号通知。"""
        if not self._enabled:
            return
        o = signal.ocean_profile
        text = (
            f"*{signal.agent_name}* "
            f"(O{o.get('openness',0)}/C{o.get('conscientiousness',0)}/"
            f"E{o.get('extraversion',0)}/A{o.get('agreeableness',0)}/"
            f"N{o.get('neuroticism',0)})\n"
            f"{signal.action.value} {signal.asset} @ ${signal.entry_price:,.2f}\n"
            f"Size: {signal.size_pct}% | "
            f"SL: ${signal.stop_loss_price or 0:,.2f} | "
            f"TP: ${signal.take_profit_price or 0:,.2f}\n"
            f"Confidence: {signal.confidence:.2f}\n"
            f"_{signal.reasoning}_\n"
            f"_{signal.personality_influence}_"
        )
        await self.send_message(text)

    async def notify_stop_loss(
        self, agent_name: str, asset: str, loss_pct: float
    ) -> None:
        """推送止损触发通知。"""
        if not self._enabled:
            return
        text = f"*{agent_name}* {asset} 止损触发，亏损 {loss_pct:.2f}%"
        await self.send_message(text)

    async def notify_daily_report(self, leaderboard: list[dict]) -> None:
        """推送每日汇总报告（排行榜表格）。"""
        if not self._enabled:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        lines: list[str] = [f"*Daily Report - {today}*\n"]
        lines.append("| # | Agent | PnL | Sharpe | MaxDD | Trades |")
        lines.append("|---|-------|-----|--------|-------|--------|")
        for idx, row in enumerate(leaderboard, 1):
            lines.append(
                f"| {idx} | {row.get('agent_name', '?')} "
                f"| ${row.get('pnl', 0):+,.0f} "
                f"| {row.get('sharpe', 0):.2f} "
                f"| {row.get('max_drawdown', 0):.1f}% "
                f"| {row.get('trades', 0)} |"
            )
        await self.send_message("\n".join(lines))

    async def notify_reflection(self, agent_name: str, reflection: dict) -> None:
        """推送 Agent 反思报告。"""
        if not self._enabled:
            return
        summary = reflection.get("summary", "无摘要")
        state = reflection.get("emotional_state", "未知")
        text = f"*{agent_name} 反思报告*\n{summary}\n情绪状态: {state}"
        await self.send_message(text)

    async def notify_risk_halt(self, reason: str) -> None:
        """推送风控暂停告警。"""
        if not self._enabled:
            return
        await self.send_message(f"*风控告警* 系统暂停交易\n原因: {reason}")

    async def notify_drift_alert(self, agent_name: str, drift_info: dict) -> None:
        """推送行为漂移告警。"""
        if not self._enabled:
            return
        severity = drift_info.get("severity", "unknown")
        reasons = "\n".join(drift_info.get("alert_reasons", []))
        await self.send_message(
            f"*行为漂移告警* [{severity.upper()}]\n"
            f"Agent: {agent_name}\n{reasons}"
        )

    async def notify_cost_report(
        self, agent_id: str, total_cost: float, breakdown: dict
    ) -> None:
        """推送成本报告。"""
        if not self._enabled:
            return
        await self.send_message(
            f"*成本报告* {agent_id}\n"
            f"总成本: ${total_cost:,.2f}\n"
            f"滑点: ${breakdown.get('slippage', 0):,.2f} | "
            f"手续费: ${breakdown.get('fees', 0):,.2f}"
        )

    async def close(self) -> None:
        """关闭 Bot session。"""
        if self._bot is not None:
            try:
                await self._bot.session.close()
            except Exception as exc:
                logger.error("关闭 Telegram Bot session 失败: {}", exc)
            self._bot = None
            logger.info("Telegram Bot session 已关闭")
