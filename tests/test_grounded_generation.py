from technical_knowledge_assistant.config import Settings
from technical_knowledge_assistant.generation.grounded import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    GroundedAnswerGenerator,
)
from technical_knowledge_assistant.models import CorpusChunk, RetrievedChunk


def _settings(**values: object) -> Settings:
    return Settings(_env_file=None, llm_provider="none", **values)


def test_refuses_when_evidence_has_only_one_meaningful_question_term() -> None:
    answer = GroundedAnswerGenerator(_settings()).answer(
        "What is the capital of Iceland?",
        [RetrievedChunk(CorpusChunk("capital-example", "The capital of the United States is Washington."), 1.0)],
    )

    assert answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert answer.citations == ()
    assert not answer.sufficient_evidence


def test_uses_evidence_with_two_meaningful_question_terms() -> None:
    answer = GroundedAnswerGenerator(_settings()).answer(
        "How do I invoke a runnable?",
        [RetrievedChunk(CorpusChunk("runnable-doc", "Invoke a runnable with invoke()."), 1.0)],
    )

    assert answer.sufficient_evidence
    assert answer.citations == ("runnable-doc",)


def test_context_selection_caps_total_and_chunk_characters() -> None:
    generator = GroundedAnswerGenerator(
        _settings(max_context_characters=20, max_chunk_characters=10, min_evidence_term_overlap=1)
    )
    selected = generator._select_evidence(
        "Runnable invoke",
        [RetrievedChunk(CorpusChunk("chunk", "Runnable invoke " * 20), 1.0)],
    )

    assert len(selected) == 1
    assert len(selected[0].chunk.text) == 10