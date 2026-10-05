"""Execute retrieval evaluation and persist reproducible JSON results."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from technical_knowledge_assistant.evaluation.freshstack import EvaluationQuery
from technical_knowledge_assistant.evaluation.metrics import (
    AnswerMetrics,
    RetrievalMetrics,
    evaluate_rankings,
    nugget_is_covered,
)
from technical_knowledge_assistant.generation.grounded import CITATION_PATTERN
from technical_knowledge_assistant.models import Answer
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


@dataclass(frozen=True)
class QueryAnswerResult:
    """Generated answer and offline quality signals for one evaluation query."""

    query_id: str
    answer_text: str
    selected_citations: list[str]
    inline_citations: list[str]
    refused: bool
    nugget_count: int
    covered_nugget_count: int
    citations_valid: bool


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """Aggregate answer quality and per-query records for one fixed partition."""

    partition: str
    metrics: AnswerMetrics
    queries: list[QueryAnswerResult]


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


def evaluate_answers(
    answer_question: Callable[[str], Answer],
    queries: list[EvaluationQuery],
    partition: str,
    nugget_threshold: float = 0.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AnswerEvaluationResult:
    """Generate answers, then score them using evaluation-only nuggets and judgments."""
    results: list[QueryAnswerResult] = []
    total_queries = len(queries)
    for completed_queries, query in enumerate(queries, start=1):
        answer = answer_question(query.text)
        inline_citations = list(dict.fromkeys(CITATION_PATTERN.findall(answer.text)))
        selected_citations = list(answer.citations)
        covered_nugget_count = sum(
            nugget_is_covered(answer.text, nugget, threshold=nugget_threshold)
            for nugget in query.nuggets
        )
        results.append(
            QueryAnswerResult(
                query_id=query.query_id,
                answer_text=answer.text,
                selected_citations=selected_citations,
                inline_citations=inline_citations,
                refused=not answer.sufficient_evidence,
                nugget_count=len(query.nuggets),
                covered_nugget_count=covered_nugget_count,
                citations_valid=set(inline_citations).issubset(selected_citations),
            )
        )
        if progress_callback and (completed_queries % 10 == 0 or completed_queries == total_queries):
            progress_callback(completed_queries, total_queries)

    answered_results = [result for result in results if not result.refused]
    total_nuggets = sum(result.nugget_count for result in results)
    covered_nuggets = sum(result.covered_nugget_count for result in results)
    supported_results = [
        result for query, result in zip(queries, results, strict=True) if query.relevant_chunk_ids
    ]
    unsupported_results = [
        result for query, result in zip(queries, results, strict=True) if not query.relevant_chunk_ids
    ]
    metrics = AnswerMetrics(
        query_count=len(results),
        nugget_coverage=covered_nuggets / total_nuggets if total_nuggets else 0.0,
        citation_presence_rate=(
            sum(bool(result.inline_citations) for result in answered_results) / len(answered_results)
            if answered_results
            else 0.0
        ),
        citation_validity_rate=(
            sum(result.citations_valid for result in answered_results) / len(answered_results)
            if answered_results
            else 0.0
        ),
        supported_refusal_rate=(
            sum(result.refused for result in supported_results) / len(supported_results)
            if supported_results
            else 0.0
        ),
        unsupported_refusal_rate=(
            sum(result.refused for result in unsupported_results) / len(unsupported_results)
            if unsupported_results
            else 0.0
        ),
    )
    return AnswerEvaluationResult(partition=partition, metrics=metrics, queries=results)


def save_answer_result(result: AnswerEvaluationResult, directory: Path) -> Path:
    """Persist one reproducible answer-evaluation artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"answers-{result.partition}-{result.metrics.query_count}.json"
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return output_path