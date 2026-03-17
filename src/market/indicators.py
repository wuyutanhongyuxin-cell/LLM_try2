from __future__ import annotations

"""
技术指标计算模块。

纯 Python 实现 RSI、SMA、MACD 三个常用技术指标。
禁止使用 pandas / numpy，所有计算基于内置类型。
"""


def calculate_sma(prices: list[float], period: int = 20) -> float | None:
    """计算简单移动平均线 (SMA)。数据不足时返回 None。"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def _calculate_ema(prices: list[float], period: int) -> float | None:
    """计算指数移动平均线 (EMA)。用前 period 条 SMA 作初始值。"""
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


def calculate_rsi(prices: list[float], period: int = 14) -> float | None:
    """计算相对强弱指数 (RSI)。需要至少 period+1 条数据。

    公式：RSI = 100 - 100 / (1 + RS)，RS = 平均涨幅 / 平均跌幅。
    """
    if len(prices) < period + 1:
        return None
    # 取最近 period 个价格变动
    changes = [prices[i] - prices[i - 1] for i in range(len(prices) - period, len(prices))]
    gains = sum(c for c in changes if c > 0) / period
    losses = sum(-c for c in changes if c < 0) / period
    if losses == 0:
        return 100.0  # 全部上涨
    return round(100.0 - 100.0 / (1.0 + gains / losses), 4)


def calculate_macd(
    prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9,
) -> dict[str, float] | None:
    """计算 MACD 指标。

    返回 {"macd": float, "signal": float, "histogram": float}。
    数据不足时返回 None。
    """
    if len(prices) < slow + signal:
        return None
    k_fast, k_slow = 2.0 / (fast + 1), 2.0 / (slow + 1)
    # 初始化两条 EMA
    ema_fast = sum(prices[:fast]) / fast
    ema_slow = sum(prices[:slow]) / slow
    # 把 fast EMA 推进到 slow 起点
    for p in prices[fast:slow]:
        ema_fast = p * k_fast + ema_fast * (1 - k_fast)
    # 从 slow 位置开始收集 MACD 序列
    macd_series: list[float] = [ema_fast - ema_slow]
    for p in prices[slow:]:
        ema_fast = p * k_fast + ema_fast * (1 - k_fast)
        ema_slow = p * k_slow + ema_slow * (1 - k_slow)
        macd_series.append(ema_fast - ema_slow)
    if len(macd_series) < signal:
        return None
    # 信号线 = MACD 序列的 EMA
    signal_val = _calculate_ema(macd_series, signal)
    if signal_val is None:
        return None
    macd_val = macd_series[-1]
    return {
        "macd": round(macd_val, 4),
        "signal": round(signal_val, 4),
        "histogram": round(macd_val - signal_val, 4),
    }
