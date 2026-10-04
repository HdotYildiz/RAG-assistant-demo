from technical_knowledge_assistant.data.freshstack import normalize_chunk


def test_normalize_chunk_prefers_stable_ids_and_text() -> None:
    chunk = normalize_chunk(
        {
            "chunk_id": "langchain-42",
            "document_id": "docs/chains.md",
            "text": "A runnable can be invoked with invoke().",
            "language": "python",
        },
        position=4,
    )

    assert chunk.chunk_id == "langchain-42"
    assert chunk.document_id == "docs/chains.md"
    assert chunk.text == "A runnable can be invoked with invoke()."
    assert chunk.metadata == {"language": "python"}


def test_normalize_chunk_uses_freshstack_document_id() -> None:
    chunk = normalize_chunk({"_id": "freshstack-doc-7", "text": "Corpus content."}, position=7)

    assert chunk.chunk_id == "freshstack-doc-7"