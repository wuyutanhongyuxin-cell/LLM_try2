from __future__ import annotations

"""交易成本模型：滑点 + 手续费 + 资金费率。

支持两种市场的成本计算：
- 加密货币永续合约：滑点(bps) + 百分比手续费 + 资金费率
- CME 期货：滑点(bps) + 按手佣金，无资金费率
"""

from pydantic import BaseModel, Field


class CostConfig(BaseModel):
    """加密货币永续合约成本配置（百分比费率）"""

    slippage_bps: float = Field(default=5.0, description="滑点 basis points，5 = 0.05%")
    taker_fee_rate: float = Field(default=0.0004, description="Taker 手续费率 0.04%")
    maker_fee_rate: float = Field(default=0.0002, description="Maker 手续费率 0.02%")
    funding_rate_8h: float = Field(
        default=0.00015, description="8h资金费率 0.015%（2024 BTC-USDT 均值约 0.017%）"
    )
    enable_costs: bool = Field(default=True, description="是否启用成本计算")


class CMECostConfig(BaseModel):
    """CME 期货成本配置（按手佣金，无资金费率）"""

    slippage_bps: float = Field(default=2.0, description="滑点 basis points，2 = 0.02%")
    commission_per_contract: float = Field(
        default=1.25, description="单边佣金（美元/手），含交易所+清算费"
    )
    enable_costs: bool = Field(default=True, description="是否启用成本计算")


class CostResult(BaseModel):
    """单次交易的成本明细"""

    slippage_cost: float = Field(..., description="滑点成本（绝对值）")
    fee_cost: float = Field(..., description="手续费（绝对值）")
    total_cost: float = Field(..., description="总成本")
    effective_price: float = Field(..., description="考虑滑点后的实际成交价")


def calculate_entry_cost(
    price: float, notional: float, side: str, config: CostConfig
) -> CostResult:
    """计算开仓成本。

    - side="LONG": 滑点抬高成交价（买入时价格更高）
    - side="SHORT": 滑点压低成交价（卖出时价格更低）
    - enable_costs=False 时返回零成本
    """
    # 成本计算关闭时，返回零成本结果
    if not config.enable_costs:
        return CostResult(
            slippage_cost=0.0, fee_cost=0.0, total_cost=0.0, effective_price=price
        )

    # 滑点百分比：basis points 转小数
    slip_pct = config.slippage_bps / 10000.0

    # 根据方向计算实际成交价
    if side == "LONG":
        effective = price * (1 + slip_pct)
    else:
        effective = price * (1 - slip_pct)

    # 滑点成本 = 价差 * 数量（notional / price 得到数量）
    slippage_cost = abs(effective - price) * (notional / price)
    # 手续费 = 名义价值 * taker费率
    fee_cost = notional * config.taker_fee_rate

    return CostResult(
        slippage_cost=round(slippage_cost, 6),
        fee_cost=round(fee_cost, 6),
        total_cost=round(slippage_cost + fee_cost, 6),
        effective_price=round(effective, 6),
    )


def calculate_exit_cost(
    price: float, notional: float, side: str, config: CostConfig
) -> CostResult:
    """计算平仓成本（方向与开仓相反）。"""
    exit_side = "SHORT" if side == "LONG" else "LONG"
    return calculate_entry_cost(price, notional, exit_side, config)


def calculate_funding_cost(
    notional: float, holding_hours: float, config: CostConfig
) -> float:
    """计算持仓期间累计资金费率成本（仅加密货币永续合约）。

    每 8 小时结算一次：funding_periods = holding_hours / 8
    cost = notional * funding_rate_8h * funding_periods
    """
    if not config.enable_costs:
        return 0.0
    periods = holding_hours / 8.0
    return round(notional * config.funding_rate_8h * periods, 6)


# ── CME 期货成本计算 ──────────────────────────────────────


def calculate_cme_entry_cost(
    price: float,
    contracts: int,
    multiplier: float,
    side: str,
    config: CMECostConfig,
) -> CostResult:
    """计算 CME 期货开仓成本。

    Args:
        price: 合约价格
        contracts: 手数
        multiplier: 合约乘数（如 ES=50, CL=1000）
        side: "LONG" 或 "SHORT"
        config: CME 成本配置
    """
    if not config.enable_costs:
        return CostResult(
            slippage_cost=0.0, fee_cost=0.0, total_cost=0.0, effective_price=price
        )
    notional = price * contracts * multiplier
    slip_pct = config.slippage_bps / 10000.0
    if side == "LONG":
        effective = price * (1 + slip_pct)
    else:
        effective = price * (1 - slip_pct)
    slippage_cost = abs(effective - price) * contracts * multiplier
    fee_cost = config.commission_per_contract * contracts
    return CostResult(
        slippage_cost=round(slippage_cost, 6),
        fee_cost=round(fee_cost, 6),
        total_cost=round(slippage_cost + fee_cost, 6),
        effective_price=round(effective, 6),
    )


def calculate_cme_exit_cost(
    price: float,
    contracts: int,
    multiplier: float,
    side: str,
    config: CMECostConfig,
) -> CostResult:
    """计算 CME 期货平仓成本（方向与开仓相反）。"""
    exit_side = "SHORT" if side == "LONG" else "LONG"
    return calculate_cme_entry_cost(price, contracts, multiplier, exit_side, config)
