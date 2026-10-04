"""Build and persist BM25 and dense indexes over normalized corpus chunks."""

import json
import pickle
import re
from dataclasses import asdict
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from technical_knowledge_assistant.models import CorpusChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize code identifiers and natural language consistently for BM25."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HybridIndex:
    """A loaded local hybrid index and its source chunk mapping."""

    def __init__(
        self,
        chunks: list[CorpusChunk],
        bm25: BM25Okapi,
        dense_index: faiss.Index,
        embedding_model: str,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.dense_index = dense_index
        self.embedding_model = embedding_model

    def save(self, directory: Path) -> None:
        """Persist the index and chunk metadata in a portable local directory."""
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as output:
            for chunk in self.chunks:
                output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        with (directory / "bm25.pkl").open("wb") as output:
            pickle.dump(self.bm25, output)
        faiss.write_index(self.dense_index, str(directory / "dense.faiss"))
        (directory / "manifest.json").write_text(
            json.dumps({"embedding_model": self.embedding_model, "chunk_count": len(self.chunks)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        """Load a previously built index, failing clearly when it is incomplete."""
        required_files = ("chunks.jsonl", "bm25.pkl", "dense.faiss", "manifest.json")
        missing = [name for name in required_files if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"No complete index in {directory}. Missing: {', '.join(missing)}. Run `rag-assistant index`."
            )
        with (directory / "chunks.jsonl").open(encoding="utf-8") as source:
            chunks = [CorpusChunk(**json.loads(line)) for line in source if line.strip()]
        with (directory / "bm25.pkl").open("rb") as source:
            bm25 = pickle.load(source)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            chunks=chunks,
            bm25=bm25,
            dense_index=faiss.read_index(str(directory / "dense.faiss")),
            embedding_model=manifest["embedding_model"],
        )


def build_index(chunks: list[CorpusChunk], embedding_model: str) -> HybridIndex:
    """Create a hybrid index from corpus chunks, with normalized dense vectors."""
    if not chunks:
        raise ValueError("Cannot build an index from an empty corpus.")
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    dense_index = faiss.IndexFlatIP(embeddings.shape[1])
    dense_index.add(embeddings)
    return HybridIndex(
        chunks=chunks,
        bm25=BM25Okapi([tokenize(chunk.text) for chunk in chunks]),
        dense_index=dense_index,
        embedding_model=embedding_model,
    )