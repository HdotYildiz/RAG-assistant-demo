"""Metrics for retrieval, answer coverage, citations, and refusals."""

import re
from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RetrievalMetrics:
    """Mean retrieval metrics over evaluated queries."""

    query_count: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


@dataclass(frozen=True)
class AnswerMetrics:
    """Aggregate quality signals for answers evaluated against FreshStack nuggets."""

    query_count: int
    nugget_coverage: float
    citation_presence_rate: float
    citation_validity_rate: float
    supported_refusal_rate: float
    unsupported_refusal_rate: float


EVALUATION_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "to", "with",
    }
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


def content_terms(text: str) -> set[str]:
    """Normalize evaluation text into non-trivial terms for transparent lexical scoring."""
    return {
        term.lower()
        for term in WORD_PATTERN.findall(text)
        if len(term) >= 3 and term.lower() not in EVALUATION_STOPWORDS
    }


def nugget_is_covered(answer_text: str, nugget: str, threshold: float = 0.5) -> bool:
    """Treat a nugget as covered when enough of its meaningful terms appear in the answer."""
    nugget_terms = content_terms(nugget)
    if not nugget_terms:
        return False
    return len(nugget_terms & content_terms(answer_text)) / len(nugget_terms) >= threshold


def evaluate_rankings(
    rankings: list[tuple[list[str], frozenset[str]]],
    k: int,
) -> RetrievalMetrics:
    """Calculate mean Recall@k, MRR@k, and nDCG@k from ranked chunk IDs."""
    if k < 1:
        raise ValueError("k must be positive.")
    scored_rankings = [(ranked_ids[:k], relevant_ids) for ranked_ids, relevant_ids in rankings if relevant_ids]
    if not scored_rankings:
        return RetrievalMetrics(query_count=0, recall_at_k=0.0, mrr_at_k=0.0, ndcg_at_k=0.0)

    recall_total = 0.0
    reciprocal_rank_total = 0.0
    ndcg_total = 0.0
    for ranked_ids, relevant_ids in scored_rankings:
        relevance = [chunk_id in relevant_ids for chunk_id in ranked_ids]
        recall_total += sum(relevance) / len(relevant_ids)
        reciprocal_rank_total += next(
            (1 / rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
            0.0,
        )
        dcg = sum(1 / log2(rank + 1) for rank, is_relevant in enumerate(relevance, start=1) if is_relevant)
        ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1))
        ndcg_total += dcg / ideal_dcg

    query_count = len(scored_rankings)
    return RetrievalMetrics(
        query_count=query_count,
        recall_at_k=recall_total / query_count,
        mrr_at_k=reciprocal_rank_total / query_count,
        ndcg_at_k=ndcg_total / query_count,
    )