import pytest

from technical_knowledge_assistant.evaluation.freshstack import normalize_query, split_queries
from technical_knowledge_assistant.evaluation.metrics import evaluate_rankings


def test_normalize_query_unions_nugget_relevance_ids() -> None:
    query = normalize_query(
        {
            "query_id": 12,
            "query_title": "Runnable invocation",
            "query_text": "How do I invoke it?",
            "nuggets": [
                {"relevant_corpus_ids": ["chunk-1", "chunk-2"]},
                {"relevant_corpus_ids": ["chunk-2", "chunk-3"]},
            ],
        }
    )

    assert query.query_id == "12"
    assert query.text == "Runnable invocation How do I invoke it?"
    assert query.relevant_chunk_ids == frozenset({"chunk-1", "chunk-2", "chunk-3"})


def test_split_queries_is_reproducible() -> None:
    queries = [normalize_query({"query_id": index, "query_title": str(index)}) for index in range(10)]

    first_development, first_final = split_queries(queries, development_fraction=0.7, seed=42)
    second_development, second_final = split_queries(queries, development_fraction=0.7, seed=42)

    assert first_development == second_development
    assert first_final == second_final
    assert len(first_development) == 7
    assert len(first_final) == 3


def test_evaluate_rankings_calculates_binary_metrics() -> None:
    metrics = evaluate_rankings(
        [
            (["wrong", "relevant-1", "relevant-2"], frozenset({"relevant-1", "relevant-2"})),
            (["wrong"], frozenset({"relevant-3"})),
        ],
        k=3,
    )

    assert metrics.query_count == 2
    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.mrr_at_k == pytest.approx(0.25)
    assert metrics.ndcg_at_k == pytest.approx(0.3467, abs=0.0001)