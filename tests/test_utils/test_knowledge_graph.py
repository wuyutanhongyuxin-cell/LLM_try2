from __future__ import annotations

"""知识图谱加载与查询测试。"""

from src.utils.knowledge_graph import (
    build_knowledge_context,
    get_causal_factors,
    load_graph,
)


class TestLoadGraph:
    """图谱加载测试。"""

    def test_returns_dict_with_required_keys(self) -> None:
        """load_graph() 返回包含 entities 和 causal_relations 的 dict。"""
        graph = load_graph()
        assert isinstance(graph, dict)
        assert "entities" in graph
        assert "causal_relations" in graph

    def test_cache_returns_same_object(self) -> None:
        """两次调用 load_graph() 应返回同一对象（缓存生效）。"""
        g1 = load_graph()
        g2 = load_graph()
        assert g1 is g2


class TestGetCausalFactors:
    """因果因子查询测试。"""

    def test_btc_has_factors(self) -> None:
        """BTC 应有非空因果因子，且第一个为 strong。"""
        factors = get_causal_factors("BTC")
        assert len(factors) > 0
        assert factors[0]["strength"] == "strong"

    def test_unknown_asset_returns_empty(self) -> None:
        """查询不存在的资产应返回空列表，不报错。"""
        factors = get_causal_factors("UNKNOWN_ASSET_XYZ")
        assert factors == []


class TestBuildKnowledgeContext:
    """知识上下文构建测试。"""

    def test_contains_market_knowledge_header(self) -> None:
        """构建结果包含 MARKET KNOWLEDGE 标题。"""
        ctx = build_knowledge_context("BTC")
        assert "MARKET KNOWLEDGE" in ctx

    def test_contains_key_factors(self) -> None:
        """BTC 上下文应包含 M2 和 FED 关键词。"""
        ctx = build_knowledge_context("BTC")
        assert "M2" in ctx
        assert "FED" in ctx
