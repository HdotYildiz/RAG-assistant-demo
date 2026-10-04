"""Generate answers strictly from retrieved chunks."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from technical_knowledge_assistant.config import Settings
from technical_knowledge_assistant.models import Answer, RetrievedChunk


class GroundedAnswerGenerator:
    """Answer from supplied evidence using optional OpenAI-compatible providers."""

    def __init__(self, runtime_settings: Settings) -> None:
        self.settings = runtime_settings

    def answer(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        """Return a cited answer or explicitly state when no corpus evidence was retrieved."""
        if not evidence:
            return Answer(
                text="I do not have enough information in the indexed knowledge source to answer that.",
                citations=(),
                sufficient_evidence=False,
            )
        if self.settings.llm_provider in {"ollama", "openai"}:
            text = self._generate_chat_completion(question, evidence)
        else:
            text = self._extractive_answer(evidence)
        citations = tuple(chunk.chunk.chunk_id for chunk in evidence)
        return Answer(text=text, citations=citations, sufficient_evidence=True)

    @staticmethod
    def _extractive_answer(evidence: list[RetrievedChunk]) -> str:
        excerpts = [f"[{item.chunk.chunk_id}] {item.chunk.text}" for item in evidence[:3]]
        return "The indexed corpus provides these relevant excerpts:\n\n" + "\n\n".join(excerpts)

    def _generate_chat_completion(self, question: str, evidence: list[RetrievedChunk]) -> str:
        """Call the selected provider through the OpenAI Chat Completions contract."""
        context = "\n\n".join(f"[{item.chunk.chunk_id}]\n{item.chunk.text}" for item in evidence)
        prompt = (
            "Answer using only the supplied corpus chunks. Cite every factual claim with its "
            "chunk ID in square brackets. If the chunks do not establish the answer, say that "
            "the knowledge source does not contain enough information.\n\n"
            f"Question: {question}\n\nCorpus chunks:\n{context}"
        )
        payload = json.dumps(
            {
                "model": self.settings.llm_model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        base_url = (
            self.settings.ollama_base_url
            if self.settings.llm_provider == "ollama"
            else self.settings.openai_base_url
        )
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_provider == "openai":
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
        request = Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"{self.settings.llm_provider} generation request failed: {error}") from error
        return response_data["choices"][0]["message"]["content"].strip()