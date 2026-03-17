"""生成 CME 期货合成历史数据 CSV。

用法: python scripts/generate_cme_data.py
输出: data/es_1h_2024.csv, data/nq_1h_2024.csv, data/cl_1h_2024.csv 等

基于真实的价格区间和波动率特征生成逼真的模拟 OHLCV 数据。
"""
from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# CME 合约特征：(起始价, 日波动率%, 日均成交量)
_CONTRACTS: dict[str, tuple[float, float, int]] = {
    "ES": (5200.0, 0.8, 1_500_000),
    "NQ": (18000.0, 1.1, 600_000),
    "CL": (72.0, 1.8, 400_000),
    "GC": (2400.0, 0.7, 250_000),
    "SI": (28.0, 1.5, 80_000),
    "ZB": (118.0, 0.4, 300_000),
}

# 生成参数
_BARS = 2000          # 约 83 天的 1h K 线
_HOURS_PER_DAY = 23   # CME 几乎 23h 交易


def _generate_ohlcv(
    symbol: str,
    start_price: float,
    daily_vol_pct: float,
    avg_volume: int,
    bars: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """生成一组 OHLCV 数据。使用几何布朗运动 + 均值回归模拟。"""
    rng = random.Random(seed)
    hourly_vol = daily_vol_pct / 100 / math.sqrt(_HOURS_PER_DAY)
    # 添加一个缓慢的趋势（随机选牛/熊/震荡）
    trend_type = rng.choice(["bull", "bear", "sideways"])
    daily_drift = {"bull": 0.0003, "bear": -0.0002, "sideways": 0.0}[trend_type]
    hourly_drift = daily_drift / _HOURS_PER_DAY

    rows: list[dict[str, str]] = []
    price = start_price
    ts = datetime(2024, 1, 2, 18, 0)  # 从 2024-01-02 18:00 开始

    for i in range(bars):
        # 几何布朗运动
        shock = rng.gauss(0, 1)
        ret = hourly_drift + hourly_vol * shock
        # 每 200 根 K 线可能出现一次 regime 切换
        if i > 0 and i % 200 == 0:
            trend_type = rng.choice(["bull", "bear", "sideways"])
            daily_drift = {"bull": 0.0003, "bear": -0.0002, "sideways": 0.0}[trend_type]
            hourly_drift = daily_drift / _HOURS_PER_DAY

        open_p = price
        close_p = price * (1 + ret)
        # 生成逼真的 high/low
        intra_range = abs(ret) + hourly_vol * abs(rng.gauss(0, 0.5))
        high_p = max(open_p, close_p) * (1 + intra_range * 0.5)
        low_p = min(open_p, close_p) * (1 - intra_range * 0.5)
        volume = max(1000, int(avg_volume / _HOURS_PER_DAY * rng.uniform(0.3, 2.5)))

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": f"{open_p:.2f}",
            "high": f"{high_p:.2f}",
            "low": f"{low_p:.2f}",
            "close": f"{close_p:.2f}",
            "volume": str(volume),
        })
        price = close_p
        ts += timedelta(hours=1)
        # 跳过 CME 休市（周六 ~16:00 到周日 ~17:00）
        if ts.weekday() == 5 and ts.hour >= 16:
            ts += timedelta(hours=25)

    return rows


def _write_csv(rows: list[dict[str, str]], path: str) -> None:
    """写入 CSV 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    first_p = rows[0]["close"]
    last_p = rows[-1]["close"]
    print(f"  {path}: {len(rows)} 行, {first_p} → {last_p}")


def main() -> None:
    """为每个 CME 合约生成合成数据。"""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    print("生成 CME 合成数据...")
    for symbol, (start_price, vol, avg_vol) in _CONTRACTS.items():
        rows = _generate_ohlcv(symbol, start_price, vol, avg_vol, _BARS, seed=hash(symbol) % 10000)
        _write_csv(rows, str(data_dir / f"{symbol.lower()}_1h_2024.csv"))
    print("完成！")


if __name__ == "__main__":
    main()
