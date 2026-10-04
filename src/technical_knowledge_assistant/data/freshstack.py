"""Load and normalize FreshStack corpus records without exposing evaluation data."""

from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset

from technical_knowledge_assistant.models import CorpusChunk

TEXT_FIELDS = ("text", "content", "chunk", "passage", "body")
ID_FIELDS = ("_id", "chunk_id", "id", "document_id", "doc_id")
DOCUMENT_ID_FIELDS = ("document_id", "doc_id", "repository", "repo_name", "url")


def _first_string(record: Mapping[str, Any], field_names: tuple[str, ...]) -> str | None:
    for field_name in field_names:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_chunk(record: Mapping[str, Any], position: int) -> CorpusChunk:
    """Convert a dataset record to the stable chunk representation used by the app."""
    text = _first_string(record, TEXT_FIELDS)
    if text is None:
        raise ValueError(
            f"Corpus record {position} has no text field. Expected one of: {', '.join(TEXT_FIELDS)}."
        )

    chunk_id = _first_string(record, ID_FIELDS) or f"row-{position}"
    document_id = _first_string(record, DOCUMENT_ID_FIELDS)
    metadata = {
        key: str(value)
        for key, value in record.items()
        if key not in {*TEXT_FIELDS, *ID_FIELDS} and value is not None and not isinstance(value, (dict, list))
    }
    return CorpusChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        metadata=metadata,
    )


def load_corpus(
    dataset_name: str,
    subset: str,
    split: str,
) -> list[CorpusChunk]:
    """Load only the designated knowledge corpus and normalize all its records."""
    dataset: Iterable[Mapping[str, Any]] = load_dataset(dataset_name, subset, split=split)
    return [normalize_chunk(record, position) for position, record in enumerate(dataset)]