"""Load only the evaluation fields needed for offline retrieval measurement."""

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class EvaluationQuery:
    """A FreshStack question and the corpus chunks judged relevant to its nuggets."""

    query_id: str
    text: str
    relevant_chunk_ids: frozenset[str]


def normalize_query(record: Mapping[str, Any]) -> EvaluationQuery:
    """Convert a FreshStack query record to retrieval-only evaluation input."""
    query_id = str(record["query_id"])
    text = " ".join(
        part.strip()
        for part in (str(record.get("query_title", "")), str(record.get("query_text", "")))
        if part.strip()
    )
    if not text:
        raise ValueError(f"Query {query_id} has no query_title or query_text.")
    relevant_chunk_ids = frozenset(
        str(chunk_id)
        for nugget in record.get("nuggets", [])
        for chunk_id in nugget.get("relevant_corpus_ids", [])
    )
    return EvaluationQuery(query_id=query_id, text=text, relevant_chunk_ids=relevant_chunk_ids)


def load_queries(dataset_name: str, subset: str, split: str = "test") -> list[EvaluationQuery]:
    """Load the query dataset for evaluation; it is never passed to the assistant."""
    dataset: Iterable[Mapping[str, Any]] = load_dataset(dataset_name, subset, split=split)
    return [normalize_query(record) for record in dataset]


def split_queries(
    queries: list[EvaluationQuery],
    development_fraction: float,
    seed: int,
) -> tuple[list[EvaluationQuery], list[EvaluationQuery]]:
    """Create deterministic development and final sets without changing query contents."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one.")
    ordered = sorted(queries, key=lambda query: query.query_id)
    random.Random(seed).shuffle(ordered)
    development_count = round(len(ordered) * development_fraction)
    return ordered[:development_count], ordered[development_count:]