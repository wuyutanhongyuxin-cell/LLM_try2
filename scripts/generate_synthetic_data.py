"""从单个历史CSV生成多市况合成数据。

用法: python scripts/generate_synthetic_data.py --csv data/btc_1h_2024.csv --output data/

输出:
  data/btc_bear.csv    — 模拟熊市（价格趋势性下跌30%）
  data/btc_sideways.csv — 模拟横盘（价格围绕均值±5%震荡）
  data/btc_bull.csv    — 原始数据（已是牛市）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="多市况合成数据生成")
    p.add_argument("--csv", required=True, help="原始历史 CSV 路径")
    p.add_argument("--output", default="data/", help="输出目录")
    return p.parse_args()


def _read_csv(path: str) -> list[dict[str, str]]:
    """读取 CSV 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, str]], path: str) -> None:
    """写入 CSV 文件。"""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  已生成: {path} ({len(rows)} 行)")


def _apply_bear(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """熊市变换：价格线性递减 30%。"""
    total = len(rows)
    result: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        factor = 1.0 - 0.30 * i / max(total - 1, 1)
        new_row = row.copy()
        for col in ("open", "high", "low", "close"):
            if col in new_row:
                new_row[col] = str(round(float(new_row[col]) * factor, 2))
        result.append(new_row)
    return result


def _apply_sideways(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """横盘变换：压缩到均价 ±5%。"""
    closes = [float(r["close"]) for r in rows]
    mean_price = sum(closes) / len(closes) if closes else 1.0
    result: list[dict[str, str]] = []
    for row in rows:
        new_row = row.copy()
        for col in ("open", "high", "low", "close"):
            if col in new_row:
                original = float(new_row[col])
                compressed = mean_price + (original - mean_price) * 0.1
                new_row[col] = str(round(compressed, 2))
        result.append(new_row)
    return result


def main() -> None:
    """入口：读取原始CSV，生成三种市况数据。"""
    args = _parse_args()
    rows = _read_csv(args.csv)
    print(f"原始数据: {len(rows)} 行")
    out = Path(args.output)
    # 牛市：直接复制
    _write_csv(rows, str(out / "btc_bull.csv"))
    # 熊市
    _write_csv(_apply_bear(rows), str(out / "btc_bear.csv"))
    # 横盘
    _write_csv(_apply_sideways(rows), str(out / "btc_sideways.csv"))
    print("完成！")


if __name__ == "__main__":
    main()
