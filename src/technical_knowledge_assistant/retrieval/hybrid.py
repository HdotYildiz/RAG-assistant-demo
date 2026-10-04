"""Reciprocal-rank fusion of sparse and dense corpus retrieval."""

from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from technical_knowledge_assistant.indexing.hybrid import HybridIndex, tokenize
from technical_knowledge_assistant.models import RetrievedChunk


class HybridRetriever:
    """Retrieve corpus evidence using BM25 and dense vector search."""

    def __init__(self, index: HybridIndex) -> None:
        self.index = index
        self.model = SentenceTransformer(index.embedding_model)

    def search(self, question: str, candidates: int = 40, top_k: int = 8) -> list[RetrievedChunk]:
        """Return deduplicated chunks ranked using reciprocal-rank fusion."""
        if not question.strip():
            return []
        limit = min(max(candidates, top_k), len(self.index.chunks))
        sparse_scores = self.index.bm25.get_scores(tokenize(question))
        sparse_order = np.argsort(sparse_scores)[::-1][:limit]
        query_embedding = self.model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        _, dense_indices = self.index.dense_index.search(query_embedding, limit)

        fused_scores: defaultdict[int, float] = defaultdict(float)
        for rank, chunk_index in enumerate(sparse_order, start=1):
            fused_scores[int(chunk_index)] += 1 / (60 + rank)
        for rank, chunk_index in enumerate(dense_indices[0], start=1):
            if chunk_index >= 0:
                fused_scores[int(chunk_index)] += 1 / (60 + rank)

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=self.index.chunks[chunk_index], score=score)
            for chunk_index, score in ranked
        ]