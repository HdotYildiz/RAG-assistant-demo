"""Stable domain contracts shared across indexing, retrieval, and generation."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusChunk:
    """A searchable FreshStack corpus chunk."""

    chunk_id: str
    text: str
    document_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk selected by retrieval, retaining its source identity and score."""

    chunk: CorpusChunk
    score: float


@dataclass(frozen=True)
class Answer:
    """A grounded answer and the corpus chunks supporting it."""

    text: str
    citations: tuple[str, ...]
    sufficient_evidence: bool