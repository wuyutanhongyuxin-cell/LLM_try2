"""从交易日志导出 LLM 微调训练数据（JSONL 格式）。

读取 trade_logger 存储在 Redis 中的全链路日志，
筛选出成功执行且有正 PnL 的交易决策，
转换为 OpenAI/Qwen 微调标准格式。

用法:
  python scripts/export_training_data.py --agent agent_calm_innovator --output data/finetune/
  python scripts/export_training_data.py --all --min-pnl 0 --output data/finetune/

输出: data/finetune/{agent_id}_train.jsonl
每行格式: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

过滤规则：
1. 必须 executed=True（实际执行的交易）
2. 可选 --min-pnl 过滤（默认 0，只保留盈利交易作为正样本）
3. 可选 --include-losses 同时导出亏损交易（标记 label=negative）
4. prompt_hash 非空（有完整 prompt 追溯）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _log_entry_to_training_sample(
    entry: dict, system_prompt: str, min_pnl: float, include_losses: bool
) -> dict | None:
    """将单条 TradeLogEntry 转换为微调训练样本。

    Args:
        entry: trade_logger 日志条目
        system_prompt: Agent 的 system prompt
        min_pnl: 最低 PnL 阈值
        include_losses: 是否包含亏损交易

    Returns:
        {"messages": [...]} 格式的字典，或 None（不满足过滤条件时）
    """
    # 过滤条件1：必须实际执行
    if not entry.get("executed", False):
        return None
    # 过滤条件2：prompt_hash 非空
    if not entry.get("prompt_hash", ""):
        return None
    # 获取校验后的信号（训练目标）
    post_clip = entry.get("post_clip_signal", {})
    if not post_clip:
        return None
    # PnL 过滤
    pnl = float(entry.get("execution_result", {}).get("pnl", 0))
    if pnl < min_pnl and not include_losses:
        return None
    # 构建 assistant 内容（校验后的信号 JSON）
    assistant_content = json.dumps(post_clip, ensure_ascii=False)
    # 构建 user 内容（行情快照摘要）
    snapshot = entry.get("market_snapshot", {})
    user_content = json.dumps(snapshot, ensure_ascii=False) if snapshot else "{}"
    sample: dict = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }
    # 亏损交易加标记
    if pnl < 0 and include_losses:
        sample["label"] = "negative"
    return sample


async def export_agent_data(
    agent_id: str,
    redis_url: str,
    system_prompt: str,
    min_pnl: float = 0.0,
    include_losses: bool = False,
) -> list[dict]:
    """导出指定 Agent 的全部可用训练数据。

    Args:
        agent_id: Agent 标识
        redis_url: Redis 连接 URL
        system_prompt: Agent system prompt
        min_pnl: 最低 PnL
        include_losses: 是否包含亏损

    Returns:
        训练样本列表
    """
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("错误: 需要 redis 包。pip install redis")
        return []
    key = f"trade_log:{agent_id}"
    try:
        client = aioredis.from_url(redis_url)
        raw_list = await client.lrange(key, 0, -1)
        await client.aclose()
    except Exception as exc:
        print(f"Redis 连接失败: {exc}")
        return []
    samples: list[dict] = []
    for raw in raw_list:
        entry = json.loads(raw)
        sample = _log_entry_to_training_sample(
            entry, system_prompt, min_pnl, include_losses,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="从交易日志导出 LLM 微调训练数据（JSONL 格式）"
    )
    parser.add_argument("--agent", type=str, help="Agent ID（单个导出）")
    parser.add_argument("--all", action="store_true", help="导出所有 Agent")
    parser.add_argument("--min-pnl", type=float, default=0.0, help="最低 PnL 阈值")
    parser.add_argument("--include-losses", action="store_true", help="包含亏损交易")
    parser.add_argument("--output", type=str, default="data/finetune/", help="输出目录")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--system-prompt", type=str, default="You are a crypto trader.")
    args = parser.parse_args()
    if not args.agent and not args.all:
        parser.print_help()
        sys.exit(0)
    # 确保输出目录存在
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = [args.agent] if args.agent else []
    if args.all:
        print("--all 模式需要手动指定 agent ID 列表（暂不支持自动发现）")
        sys.exit(1)
    for aid in agents:
        samples = asyncio.run(export_agent_data(
            aid, args.redis_url, args.system_prompt, args.min_pnl, args.include_losses,
        ))
        out_path = out_dir / f"{aid}_train.jsonl"
        ts = datetime.now(tz=timezone.utc).isoformat()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"// agent_id={aid}, exported_at={ts}, samples={len(samples)}\n")
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"导出 {len(samples)} 条样本 → {out_path}")


if __name__ == "__main__":
    main()
