from technical_knowledge_assistant.evaluation.freshstack import EvaluationQuery
from technical_knowledge_assistant.evaluation.runner import evaluate_retrieval, save_result
from technical_knowledge_assistant.models import CorpusChunk, RetrievedChunk


class StubRetriever:
    def search(self, question: str, candidates: int, top_k: int) -> list[RetrievedChunk]:
        del question, candidates, top_k
        return [RetrievedChunk(CorpusChunk("relevant-1", "Evidence."), score=0.1)]


def test_evaluate_retrieval_saves_per_query_rankings(tmp_path) -> None:
    progress_updates: list[tuple[int, int]] = []
    result = evaluate_retrieval(
        StubRetriever(),
        [EvaluationQuery("question-1", "Question", frozenset({"relevant-1"}))],
        candidates=10,
        k=3,
        partition="development",
        progress_callback=lambda completed, total: progress_updates.append((completed, total)),
    )

    output_path = save_result(result, tmp_path)

    assert result.metrics.recall_at_k == 1.0
    assert progress_updates == [(1, 1)]
    assert '"query_id": "question-1"' in output_path.read_text(encoding="utf-8")