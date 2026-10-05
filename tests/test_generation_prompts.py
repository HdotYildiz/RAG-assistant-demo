from technical_knowledge_assistant.generation.prompts import (
    INSUFFICIENT_INFORMATION_RESPONSE,
    build_grounded_answer_prompt,
)
from technical_knowledge_assistant.models import CorpusChunk, RetrievedChunk


def test_grounded_prompt_requires_citations_for_every_substantive_sentence() -> None:
    prompt = build_grounded_answer_prompt(
        "How do I invoke a runnable?",
        [RetrievedChunk(CorpusChunk("runnable-doc", "Use invoke() to run a runnable."), 1.0)],
    )

    assert "Every sentence containing a factual or technical claim" in prompt
    assert "recommendation, or code example must end with one or more chunk IDs" in prompt
    assert "[runnable-doc]" in prompt
    assert INSUFFICIENT_INFORMATION_RESPONSE in prompt