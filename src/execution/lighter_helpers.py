from __future__ import annotations

"""Lighter 辅助函数 — 下单、填单确认、账户查询。"""

import asyncio
import time
from decimal import Decimal

import lighter
from lighter import ApiClient, SignerClient
from loguru import logger

# SDK 约定：order_expiry=-1 表示 28 天到期，0 表示 IOC 立即到期
_GTC_28_DAY_EXPIRY = -1

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


async def place_tp_limit_order(
    signer: SignerClient,
    market_index: int,
    client_order_index: int,
    side: str,
    size: Decimal,
    base_mult: int,
    price_mult: int,
    tp_price: Decimal,
) -> None:
    """挂 GTC 限价单作为止盈。

    side='sell' 平多仓TP，'buy' 平空仓TP。
    使用 ORDER_TYPE_LIMIT + GTC（28天到期），reduce_only=True。
    """
    is_ask = side == "sell"
    price_int = int(tp_price * price_mult)
    if price_int < 1:
        price_int = 1
    tx_type, tx_info, tx_hash_signed, error = signer.sign_create_order(
        market_index=market_index,
        client_order_index=client_order_index,
        base_amount=int(size * base_mult),
        price=price_int,
        is_ask=is_ask,
        order_type=signer.ORDER_TYPE_LIMIT,
        time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
        reduce_only=True,
        trigger_price=0,
        order_expiry=_GTC_28_DAY_EXPIRY,  # SDK: -1 = 28天到期
    )
    if error is not None:
        raise RuntimeError(f"Lighter TP 签名错误: {error}")

    await signer.send_tx(tx_type=int(tx_type), tx_info=tx_info)


async def cancel_all_orders(
    signer: SignerClient, market_index: int,
) -> None:
    """取消所有挂单（平仓或反向开仓前清理旧TP单）。

    SDK 签名: sign_cancel_all_orders(time_in_force, timestamp_ms)
    time_in_force=1 (GTC) + timestamp_ms=0 → 取消所有 GTC 挂单。
    注意：该方法不按 market 过滤，会取消账户下所有挂单。
    """
    # time_in_force=1 (GTC), timestamp_ms=0 → 取消全部 GTC 挂单
    timestamp_ms = int(time.time() * 1000)
    tx_type, tx_info, tx_hash_signed, error = signer.sign_cancel_all_orders(
        time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
        timestamp_ms=timestamp_ms,
    )
    if error is not None:
        raise RuntimeError(f"Lighter 取消全部订单签名错误: {error}")

    resp = await signer.send_tx(tx_type=int(tx_type), tx_info=tx_info)
    logger.info("Lighter 已取消全部 GTC 挂单")


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


_LIGHTER_BASE = "https://mainnet.zklighter.elliot.ai"


async def fetch_candle_closes(
    api_client: ApiClient, market_index: int | None,
    resolution: str = "5m", count_back: int = 35,
) -> list[float]:
    """从 Lighter REST API 直接获取历史 K 线收盘价（不依赖 SDK 版本）。

    Args:
        resolution: K线周期，支持 1m/5m/15m/1h/4h/1d/1w
        count_back: 拉取条数（MACD 需 26+9=35）
    """
    import aiohttp
    url = f"{_LIGHTER_BASE}/api/v1/candles"
    params = {
        "market_id": market_index, "resolution": resolution,
        "start_timestamp": 0, "end_timestamp": int(time.time()),
        "count_back": count_back,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
        candles = data.get("c", [])
        return [float(c["c"]) for c in candles if c.get("c") is not None]
    except Exception as e:
        logger.warning(f"获取K线失败 ({resolution}): {e}")
        return []


async def fetch_24h_volume(api_client: ApiClient, market_index: int | None) -> float:
    """从 Lighter REST API 获取 24h 交易量（USD）。"""
    try:
        order_api = lighter.OrderApi(api_client)
        details = await order_api.order_book_details(market_id=market_index)
        d = details.order_book_details[0]
        vol = getattr(d, "daily_quote_token_volume", None)
        return float(vol) if vol else 0.0
    except Exception as e:
        logger.warning(f"获取24h交易量失败: {e}")
        return 0.0


async def fetch_last_price(api_client: ApiClient, market_index: int | None) -> Decimal:
    """通过 REST 获取最新成交价（平仓后备方案）。

    #15: 不再用硬编码 85000 兜底，失败返回 0 让调用方决定是否中止。
    """
    try:
        order_api = lighter.OrderApi(api_client)
        details = await order_api.order_book_details(market_id=market_index)
        d = details.order_book_details[0]
        if hasattr(d, "last_price") and d.last_price:
            return Decimal(str(d.last_price))
        logger.warning("Lighter API 未返回 last_price")
        return Decimal("0")
    except Exception as e:
        logger.error(f"获取最新价格失败: {e}")
        return Decimal("0")


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


async def query_leverage(
    api_client: ApiClient, account_index: int, market_index: int | None,
) -> int:
    """从 Lighter 账户读取指定市场的杠杆倍数。

    通过 initial_margin_fraction 反算：leverage = 100 / IMF（IMF 单位为百分比）。
    例：IMF=2.00 → 100/2=50x。无持仓时返回 0（需用配置文件兜底）。
    """
    try:
        account_api = lighter.AccountApi(api_client)
        data = await account_api.account(by="index", value=str(account_index))
        if data and data.accounts:
            for pos in data.accounts[0].positions:
                if pos.market_id == market_index:
                    imf = float(pos.initial_margin_fraction)
                    if imf > 0:
                        lev = int(100 / imf)
                        logger.info(f"Lighter 杠杆读取: IMF={imf}% → {lev}x")
                        return lev
        return 0
    except Exception as e:
        logger.warning(f"查询杠杆失败: {e}")
        return 0


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
