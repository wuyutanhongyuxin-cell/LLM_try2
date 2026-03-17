from __future__ import annotations

"""纯 Python TF-IDF 引擎测试。"""

import pytest

from src.utils.tfidf import cosine_similarity, rank_by_similarity, tokenize


class TestTokenize:
    """分词测试。"""

    def test_basic_tokenize(self) -> None:
        """基本分词：小写化 + 提取字母数字。"""
        result = tokenize("BUY BTC-PERP at 67000")
        assert result == ["buy", "btc", "perp", "at", "67000"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_special_chars_stripped(self) -> None:
        result = tokenize("RSI=75.2, MACD>0")
        assert "rsi" in result
        assert "75" in result
        assert "macd" in result


class TestCosineSimilarity:
    """余弦相似度测试。"""

    def test_identical_vectors(self) -> None:
        """相同向量相似度 ≈ 1.0。"""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        """正交向量相似度 ≈ 0.0。"""
        vec_a = {"a": 1.0, "b": 0.0}
        vec_b = {"c": 1.0, "d": 0.0}
        assert abs(cosine_similarity(vec_a, vec_b)) < 1e-6

    def test_empty_vector(self) -> None:
        """空向量相似度 = 0.0。"""
        assert cosine_similarity({}, {"a": 1.0}) == 0.0


class TestRankBySimilarity:
    """TF-IDF 排序测试。"""

    def test_btc_buy_ranks_first(self) -> None:
        """查询 'BUY BTC' 时，含 BTC BUY 的文档排最前。"""
        docs = [
            "BUY BTC rising strong momentum",
            "SELL ETH falling volume declining",
            "HOLD BTC stable sideways range",
        ]
        results = rank_by_similarity("BUY BTC", docs, top_k=3)
        assert len(results) == 3
        # 第一个结果应是 index 0（BUY BTC 文档）
        assert results[0][0] == 0

    def test_empty_documents(self) -> None:
        """空文档列表不崩溃。"""
        assert rank_by_similarity("BUY BTC", [], top_k=5) == []

    def test_top_k_limit(self) -> None:
        """返回数量不超过 top_k。"""
        docs = ["doc1 btc", "doc2 eth", "doc3 sol", "doc4 arb"]
        results = rank_by_similarity("btc", docs, top_k=2)
        assert len(results) == 2

    def test_similarity_scores_non_negative(self) -> None:
        """所有相似度分数 >= 0。"""
        docs = ["bitcoin buy signal", "ethereum sell order", "hold position stable"]
        results = rank_by_similarity("bitcoin buy", docs)
        for _, score in results:
            assert score >= 0.0
