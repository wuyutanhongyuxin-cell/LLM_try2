"""从 Databento 下载 CME 期货真实历史 OHLCV 数据。

用法:
  1. 在 .env 中设置 DATABENTO_API_KEY=db-xxx
  2. python scripts/download_cme_data.py

下载4个低相关品种约6个月的1小时K线:
  - ES (E-mini S&P 500)  — 股指
  - CL (Crude Oil)       — 能源
  - GC (Gold)            — 贵金属
  - ZB (US Treasury Bond) — 国债

数据源: CME Globex (GLBX.MDP3)
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 加载 .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 下载配置
_DATASET = "GLBX.MDP3"
_SCHEMA = "ohlcv-1h"
_START = "2025-09-15"
_END = "2026-03-15"

# 4个低相关品种，使用连续合约（calendar roll, front month）
_SYMBOLS = {
    "ES": "ES.c.0",   # E-mini S&P 500
    "CL": "CL.c.0",   # 原油
    "GC": "GC.c.0",   # 黄金
    "ZB": "ZB.c.0",   # 美国国债
}


def _check_api_key() -> str:
    """检查 DATABENTO_API_KEY 是否已设置。"""
    key = os.environ.get("DATABENTO_API_KEY", "")
    if not key:
        print("错误: 请在 .env 文件中设置 DATABENTO_API_KEY")
        print("  获取方式: https://databento.com/signup (免费注册)")
        print("  格式: DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxx")
        sys.exit(1)
    return key


def _estimate_cost() -> None:
    """估算并提示下载成本。"""
    hours_per_day = 23
    days = 180
    bars_per_symbol = hours_per_day * days  # ~4140
    total_bars = bars_per_symbol * len(_SYMBOLS)
    # OHLCV-1h 每条约 50 bytes
    est_bytes = total_bars * 50
    est_mb = est_bytes / 1_000_000
    print(f"预估数据量: {len(_SYMBOLS)} 品种 × {bars_per_symbol} 条 = {total_bars} 条")
    print(f"预估大小: ~{est_mb:.1f} MB (极小，费用几乎为零)")
    print(f"时间范围: {_START} → {_END}")
    print()


def _download_symbol(client, symbol_key: str, continuous_symbol: str) -> list[dict]:
    """下载单个品种的 OHLCV 数据。"""
    print(f"  下载 {symbol_key} ({continuous_symbol})...", end="", flush=True)
    try:
        data = client.timeseries.get_range(
            dataset=_DATASET,
            symbols=continuous_symbol,
            stype_in="continuous",
            schema=_SCHEMA,
            start=_START,
            end=_END,
        )
        df = data.to_df()
        if df.empty:
            print(" 无数据!")
            return []
        # to_df() 返回的价格可能是 fixed-point int 或已转换的 float
        # 检测：如果第一个非零 close > 1e6，说明仍是 fixed-point，需除以 1e9
        sample_closes = [float(r) for r in df["close"] if float(r) != 0]
        divisor = 1e9 if (sample_closes and sample_closes[0] > 1e6) else 1.0
        if divisor > 1:
            print(f" (fixed-point, ÷{divisor:.0e})", end="")
        rows: list[dict] = []
        for _, row in df.iterrows():
            ts = row.name if hasattr(row.name, 'strftime') else row.get("ts_event", "")
            o = float(row["open"]) / divisor
            h = float(row["high"]) / divisor
            l = float(row["low"]) / divisor
            c = float(row["close"]) / divisor
            # 跳过价格全为 0 的空 bar
            if c == 0 and o == 0:
                continue
            rows.append({
                "timestamp": str(ts)[:19],
                "open": f"{o:.2f}",
                "high": f"{h:.2f}",
                "low": f"{l:.2f}",
                "close": f"{c:.2f}",
                "volume": str(int(row["volume"])),
            })
        print(f" {len(rows)} 条 (过滤空bar后)")
        return rows
    except Exception as exc:
        print(f" 失败: {exc}")
        return []


def _write_csv(rows: list[dict], path: str) -> None:
    """写入 CSV。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    first_p, last_p = rows[0]["close"], rows[-1]["close"]
    print(f"    → {path} ({len(rows)} 行, {first_p} → {last_p})")


def main() -> None:
    """下载全部品种并保存为 CSV。"""
    import databento as db

    key = _check_api_key()
    print("=" * 60)
    print("Databento CME 历史数据下载")
    print("=" * 60)
    _estimate_cost()

    client = db.Historical(key=key)
    data_dir = Path(__file__).resolve().parent.parent / "data"
    success = 0

    for symbol_key, continuous_sym in _SYMBOLS.items():
        rows = _download_symbol(client, symbol_key, continuous_sym)
        if rows:
            out_path = str(data_dir / "cme" / "market" / f"{symbol_key.lower()}_1h_real.csv")
            _write_csv(rows, out_path)
            success += 1

    print()
    if success == len(_SYMBOLS):
        print(f"全部 {success} 个品种下载完成!")
    else:
        print(f"完成 {success}/{len(_SYMBOLS)} 个品种")
    print()
    print("接下来运行回测:")
    for sym in _SYMBOLS:
        csv_name = f"{sym.lower()}_1h_real.csv"
        print(f"  python scripts/backtest.py --market cme --asset {sym} --csv data/cme/market/{csv_name}")


if __name__ == "__main__":
    main()
