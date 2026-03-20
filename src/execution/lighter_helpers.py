from __future__ import annotations

"""Lighter 辅助函数 — 下单、填单确认、账户查询。"""

import asyncio
from decimal import Decimal

import lighter
from lighter import ApiClient, SignerClient
from loguru import logger

TERMINAL_STATUSES = {
    "FILLED", "CANCELED", "CANCELLED", "REJECTED",
    "EXPIRED", "DONE", "MATCHED", "EXECUTED",
}


IOC_SLIPPAGE_PCT = Decimal("0.002")  # 0.2% 滑点保证成交


async def place_ioc_order(
    signer: SignerClient,
    market_index: int,
    client_order_index: int,
    side: str,
    size: Decimal,
    base_mult: int,
    price_mult: int,
    current_price: Decimal,
    reduce_only: bool = False,
) -> None:
    """签名并发送 IOC 市价单。

    Lighter 要求 price >= 1，市价单也必须传一个带滑点的限价。
    买入：价格 = 当前价 × (1 + 0.2%)，保证吃单成交
    卖出：价格 = 当前价 × (1 - 0.2%)，保证吃单成交
    """
    is_ask = side == "sell"
    # 计算带滑点的价格（跟 grvt_lighter 逻辑一致）
    if is_ask:
        price = current_price * (Decimal("1") - IOC_SLIPPAGE_PCT)
    else:
        price = current_price * (Decimal("1") + IOC_SLIPPAGE_PCT)
    # 转为整数（乘以 price_multiplier）
    price_int = int(price * price_mult)
    if price_int < 1:
        price_int = 1

    tx_type, tx_info, tx_hash_signed, error = signer.sign_create_order(
        market_index=market_index,
        client_order_index=client_order_index,
        base_amount=int(size * base_mult),
        price=price_int,
        is_ask=is_ask,
        order_type=signer.ORDER_TYPE_MARKET,
        time_in_force=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=reduce_only,
        trigger_price=0,
        order_expiry=0,
    )
    if error is not None:
        raise RuntimeError(f"Lighter 签名错误: {error}")

    resp = await signer.send_tx(tx_type=int(tx_type), tx_info=tx_info)
    logger.info(
        f"Lighter IOC 已发送: {side} {size} @ {price:.2f} idx={client_order_index}"
    )


async def wait_for_fill(
    event: asyncio.Event,
    fill_results: dict[int, dict],
    client_order_index: int,
    timeout: float,
) -> dict | None:
    """等待 WS 填单确认事件。

    Args:
        event: 该订单的 asyncio.Event
        fill_results: 共享填单结果字典
        client_order_index: 订单标识
        timeout: 超时秒数

    Returns:
        填单结果 dict 或 None（超时）
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return fill_results.pop(client_order_index, None)
    except asyncio.TimeoutError:
        logger.warning(f"填单超时: idx={client_order_index}，按未成交处理")
        return None


async def fetch_last_price(api_client: ApiClient, market_index: int | None) -> Decimal:
    """通过 REST 获取最新成交价（平仓后备方案）。"""
    try:
        order_api = lighter.OrderApi(api_client)
        details = await order_api.order_book_details(market_id=market_index)
        d = details.order_book_details[0]
        return Decimal(str(d.last_price)) if hasattr(d, "last_price") else Decimal("85000")
    except Exception:
        return Decimal("85000")


async def query_position(
    api_client: ApiClient, account_index: int, market_index: int | None,
) -> Decimal:
    """查询 Lighter 账户在指定市场的仓位。"""
    try:
        account_api = lighter.AccountApi(api_client)
        data = await account_api.account(by="index", value=str(account_index))
        if data and data.accounts:
            for pos in data.accounts[0].positions:
                if pos.market_id == market_index:
                    qty = Decimal(str(pos.position))
                    sign = int(pos.sign) if hasattr(pos, "sign") and pos.sign else 1
                    return qty * sign
        return Decimal("0")
    except Exception as e:
        logger.error(f"查询仓位失败: {e}")
        return Decimal("0")


async def query_balance(api_client: ApiClient, account_index: int) -> Decimal:
    """查询 Lighter 账户可用余额。"""
    try:
        account_api = lighter.AccountApi(api_client)
        data = await account_api.account(by="index", value=str(account_index))
        if data and data.accounts:
            bal = data.accounts[0].available_balance
            return Decimal(str(bal)) if bal is not None else Decimal("0")
        return Decimal("0")
    except Exception as e:
        logger.error(f"查询余额失败: {e}")
        return Decimal("0")
