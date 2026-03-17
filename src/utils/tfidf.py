"""纯 Python TF-IDF + Cosine Similarity 实现。

禁止使用 sklearn、numpy、pandas。只用 math + collections + re。
用于 L2 记忆的语义相关性检索。
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """简单分词：小写化 + 提取字母数字 token。

    Args:
        text: 输入文本（英文为主）

    Returns:
        token 列表
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_tfidf(documents: list[str]) -> list[dict[str, float]]:
    """计算文档集的 TF-IDF 向量。

    Args:
        documents: 文档列表

    Returns:
        每个文档的 TF-IDF 向量（dict: token -> score）
    """
    if not documents:
        return []
    # 分词
    tokenized = [tokenize(doc) for doc in documents]
    n_docs = len(tokenized)
    # 计算 DF（文档频率）
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    # 计算每个文档的 TF-IDF
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf: Counter[str] = Counter(tokens)
        total = len(tokens)
        vec: dict[str, float] = {}
        for word, count in tf.items():
            tf_val = count / total
            idf_val = math.log(n_docs / (df[word] + 1))
            vec[word] = tf_val * idf_val
        vectors.append(vec)
    return vectors


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """计算两个稀疏向量的余弦相似度。

    Args:
        vec_a: 向量A
        vec_b: 向量B

    Returns:
        余弦相似度 [0.0, 1.0]
    """
    if not vec_a or not vec_b:
        return 0.0
    # 点积（只对共同 key 计算）
    common_keys = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(
    query: str,
    documents: list[str],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """将 query 与文档集做 TF-IDF 相似度排序。

    Args:
        query: 查询文本
        documents: 候选文档列表
        top_k: 返回前 K 个

    Returns:
        [(doc_index, similarity_score)] 按分数降序
    """
    if not documents:
        return []
    # 将 query 加入文档集一起计算 TF-IDF（确保 IDF 一致）
    all_docs = [query] + documents
    vectors = compute_tfidf(all_docs)
    query_vec = vectors[0]
    # 计算 query 与每个文档的相似度
    scored: list[tuple[int, float]] = []
    for i, doc_vec in enumerate(vectors[1:]):
        sim = cosine_similarity(query_vec, doc_vec)
        scored.append((i, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
