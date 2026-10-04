"""Standard binary relevance metrics for ranked retrieval results."""

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RetrievalMetrics:
    """Mean retrieval metrics over evaluated queries."""

    query_count: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


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