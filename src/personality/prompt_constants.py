from __future__ import annotations

"""Prompt 生成常量：各市场类型的 OCEAN 维度描述和角色模板。"""

# ── 维度描述映射：(全名, 高分含义, 低分含义) ──
# 按市场类型区分：crypto 和 cme 有不同的高/低分描述
TRAIT_DESC_CRYPTO: dict[str, tuple[str, str, str]] = {
    "openness": (
        "Openness",
        "explores new altcoins, novel strategies, and high-volatility assets",
        "sticks to BTC/ETH only, conservative and proven strategies",
    ),
    "conscientiousness": (
        "Conscientiousness",
        "strict stop-loss discipline, rigorous position sizing, rule-following",
        "impulsive trading, may ignore risk management rules",
    ),
    "extraversion": (
        "Extraversion",
        "follows market sentiment, momentum-chasing, trend-following",
        "contrarian, independent judgment, fades the crowd",
    ),
    "agreeableness": (
        "Agreeableness",
        "herding behavior, aligns with market consensus",
        "challenges consensus, comfortable taking the opposite side",
    ),
    "neuroticism": (
        "Neuroticism",
        "extreme loss aversion, very tight stops, frequent cutting of losers",
        "emotionally stable, can hold through drawdowns patiently",
    ),
}

TRAIT_DESC_CME: dict[str, tuple[str, str, str]] = {
    "openness": (
        "Openness",
        "trades diverse futures (energy, metals, bonds), embraces cross-asset strategies",
        "sticks to equity index futures (ES/NQ) only, conservative approach",
    ),
    "conscientiousness": (
        "Conscientiousness",
        "strict stop-loss discipline, rigorous position sizing, rule-following",
        "impulsive trading, may ignore risk management rules",
    ),
    "extraversion": (
        "Extraversion",
        "follows institutional flow, momentum-chasing, trend-following",
        "contrarian, independent judgment, fades the crowd",
    ),
    "agreeableness": (
        "Agreeableness",
        "herding behavior, aligns with market consensus and COT data",
        "challenges consensus, comfortable taking the opposite side",
    ),
    "neuroticism": (
        "Neuroticism",
        "extreme loss aversion, very tight stops, frequent cutting of losers",
        "emotionally stable, can hold through drawdowns patiently",
    ),
}

# 市场类型 → 描述映射
TRAIT_DESCS: dict[str, dict[str, tuple[str, str, str]]] = {
    "crypto": TRAIT_DESC_CRYPTO,
    "cme": TRAIT_DESC_CME,
}

# 市场类型 → 角色描述
# 品种描述映射：帮助 LLM 理解不同资产特性，减少因"不了解品种"输出 conf=0.0
ASSET_DESCRIPTIONS: dict[str, str] = {
    "ES": "E-mini S&P 500 index futures, tracks US large-cap equities",
    "NQ": "E-mini Nasdaq 100 futures, tracks US tech-heavy equities",
    "CL": "WTI Crude Oil futures, highly volatile commodity driven by OPEC and inventory data",
    "GC": "Gold futures, safe-haven asset inversely correlated with USD and rates",
    "ZB": "US Treasury Bond futures, inversely correlated with interest rates",
    "SI": "Silver futures, tracks gold with higher beta (~1.5x)",
    "BTC-PERP": "Bitcoin perpetual futures, highest crypto market cap",
    "ETH-PERP": "Ethereum perpetual futures, second largest crypto",
    "SOL-PERP": "Solana perpetual futures, high-performance L1 blockchain",
    "ARB-PERP": "Arbitrum perpetual futures, Ethereum L2 scaling solution",
    "DOGE-PERP": "Dogecoin perpetual futures, meme coin with high retail interest",
}

ROLE_TEMPLATES: dict[str, str] = {
    "crypto": (
        "You are a cryptocurrency perpetual futures trader with a distinct personality. "
        "Your trading persona is '{name}'. "
        "Your personality directly shapes how you analyze markets and make decisions."
    ),
    "cme": (
        "You are a CME futures trader with a distinct personality. "
        "You trade instruments like E-mini S&P 500 (ES), Nasdaq 100 (NQ), "
        "Crude Oil (CL), Gold (GC), and other CME Globex contracts. "
        "Your trading persona is '{name}'. "
        "Your personality directly shapes how you analyze markets and make decisions."
    ),
}
