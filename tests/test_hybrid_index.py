import faiss
from rank_bm25 import BM25Okapi

from technical_knowledge_assistant.indexing.hybrid import HybridIndex, tokenize
from technical_knowledge_assistant.models import CorpusChunk


def test_load_preserves_chunk_text_with_unicode_line_separator(tmp_path) -> None:
    chunk = CorpusChunk(chunk_id="chunk-1", text="First sentence.\u2028Second sentence.")
    index = HybridIndex(
        chunks=[chunk],
        bm25=BM25Okapi([tokenize(chunk.text)]),
        dense_index=faiss.IndexFlatIP(2),
        embedding_model="test-model",
    )
    index.save(tmp_path)

    loaded = HybridIndex.load(tmp_path)

    assert loaded.chunks == [chunk]