"""Execute retrieval evaluation and persist reproducible JSON results."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from technical_knowledge_assistant.evaluation.freshstack import EvaluationQuery
from technical_knowledge_assistant.evaluation.metrics import RetrievalMetrics, evaluate_rankings
from technical_knowledge_assistant.retrieval import HybridRetriever


@dataclass(frozen=True)
class QueryRetrievalResult:
    """Ranked corpus IDs recorded for one evaluation query."""

    query_id: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_count: int


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics and per-query rankings for one fixed evaluation partition."""

    partition: str
    k: int
    metrics: RetrievalMetrics
    queries: list[QueryRetrievalResult]


def evaluate_retrieval(
    retriever: HybridRetriever,
    queries: list[EvaluationQuery],
    candidates: int,
    k: int,
    partition: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationResult:
    """Retrieve each query and calculate metrics from FreshStack relevance judgments."""
    query_results: list[QueryRetrievalResult] = []
    total_queries = len(queries)
    for completed_queries, query in enumerate(queries, start=1):
        query_results.append(
            QueryRetrievalResult(
                query_id=query.query_id,
                retrieved_chunk_ids=[
                    result.chunk.chunk_id for result in retriever.search(query.text, candidates, k)
                ],
                relevant_chunk_count=len(query.relevant_chunk_ids),
            )
        )
        if progress_callback and (completed_queries % 10 == 0 or completed_queries == total_queries):
            progress_callback(completed_queries, total_queries)
    metrics = evaluate_rankings(
        [
            (result.retrieved_chunk_ids, query.relevant_chunk_ids)
            for query, result in zip(queries, query_results, strict=True)
        ],
        k,
    )
    return EvaluationResult(partition=partition, k=k, metrics=metrics, queries=query_results)


def save_result(result: EvaluationResult, directory: Path) -> Path:
    """Write one reproducible, human-readable evaluation result file."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"retrieval-{result.partition}.json"
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return output_path