"""Generate answers strictly from retrieved chunks."""

import json
import re
from dataclasses import replace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from technical_knowledge_assistant.config import Settings
from technical_knowledge_assistant.generation.prompts import build_grounded_answer_prompt
from technical_knowledge_assistant.models import Answer, RetrievedChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")
CITATION_PATTERN = re.compile(r"\[\[([^\]\n]+)\]\]")
EXTRACTIVE_EXCERPT_CHARACTERS = 600
STOPWORDS = frozenset(
    {
        "about",
        "and",
        "are",
        "can",
        "does",
        "for",
        "from",
        "how",
        "into",
        "is",
        "of",
        "the",
        "to",
        "what",
        "with",
    }
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I do not have enough information in the indexed LangChain knowledge source to answer that."
)
INSUFFICIENT_EVIDENCE_MARKERS = (
    "do not have enough information",
    "does not contain enough information",
    "insufficient information",
)


class GroundedAnswerGenerator:
    """Answer from supplied evidence using optional OpenAI-compatible providers."""

    def __init__(self, runtime_settings: Settings) -> None:
        self.settings = runtime_settings

    def answer(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        """Return a cited answer or explicitly state when no corpus evidence was retrieved."""
        selected_evidence = self._select_evidence(question, evidence)
        if not selected_evidence:
            return Answer(
                text=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=(),
                sufficient_evidence=False,
            )
        if self.settings.llm_provider in {"ollama", "openai"}:
            text = self._generate_chat_completion(question, selected_evidence)
            text = self._remove_unselected_citations(
                text,
                {item.chunk.chunk_id for item in selected_evidence},
            )
            if self._declares_insufficient_evidence(text):
                return Answer(
                    text=INSUFFICIENT_EVIDENCE_MESSAGE,
                    citations=(),
                    sufficient_evidence=False,
                )
            if not CITATION_PATTERN.search(text):
                text = self._extractive_answer(question, selected_evidence)
        else:
            text = self._extractive_answer(question, selected_evidence)
        citations = tuple(item.chunk.chunk_id for item in selected_evidence)
        return Answer(text=text, citations=citations, sufficient_evidence=True)

    def _select_evidence(
        self,
        question: str,
        evidence: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Keep evidence that lexically supports the question within the context budget."""
        question_terms = self._content_terms(question)
        selected: list[RetrievedChunk] = []
        remaining_characters = self.settings.max_context_characters
        for item in evidence:
            overlap = question_terms & self._content_terms(item.chunk.text)
            if len(overlap) < self.settings.min_evidence_term_overlap:
                continue
            header = f"[[{item.chunk.chunk_id}]]\n"
            available_characters = min(
                self.settings.max_chunk_characters,
                remaining_characters - len(header),
            )
            if available_characters <= 0:
                break
            text = item.chunk.text[:available_characters]
            selected.append(replace(item, chunk=replace(item.chunk, text=text)))
            remaining_characters -= len(header) + len(text) + 2
        return selected

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        """Return question-bearing terms while ignoring common natural-language glue."""
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if len(token) >= 3 and token.lower() not in STOPWORDS
        }

    @staticmethod
    def _remove_unselected_citations(text: str, selected_chunk_ids: set[str]) -> str:
        """Remove model-emitted citation IDs that were not provided as answer evidence."""
        return CITATION_PATTERN.sub(
            lambda match: match.group(0) if match.group(1) in selected_chunk_ids else "",
            text,
        )

    @staticmethod
    def _declares_insufficient_evidence(text: str) -> bool:
        """Identify the model declining to answer so it cannot continue with speculation."""
        normalized_text = text.lower()
        return any(marker in normalized_text for marker in INSUFFICIENT_EVIDENCE_MARKERS)

    @classmethod
    def _extractive_answer(cls, question: str, evidence: list[RetrievedChunk]) -> str:
        excerpts = [
            f"- {cls._focused_excerpt(question, item.chunk.text)} [[{item.chunk.chunk_id}]]"
            for item in evidence[:2]
        ]
        return "The indexed corpus provides these relevant excerpts:\n\n" + "\n\n".join(excerpts)

    @classmethod
    def _focused_excerpt(cls, question: str, text: str) -> str:
        """Return a compact source excerpt centered near the first question term match."""
        question_terms = cls._content_terms(question)
        match = next(
            (token for token in TOKEN_PATTERN.finditer(text) if token.group().lower() in question_terms),
            None,
        )
        start = max(0, match.start() - EXTRACTIVE_EXCERPT_CHARACTERS // 4) if match else 0
        end = min(len(text), start + EXTRACTIVE_EXCERPT_CHARACTERS)
        excerpt = " ".join(text[start:end].split())
        prefix = "..." if start else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{excerpt}{suffix}"

    def _generate_chat_completion(self, question: str, evidence: list[RetrievedChunk]) -> str:
        """Call the selected provider through the OpenAI Chat Completions contract."""
        prompt = build_grounded_answer_prompt(question, evidence)
        payload = json.dumps(
            {
                "model": self.settings.llm_model,
                "temperature": 0,
                "max_tokens": self.settings.generation_max_tokens,
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