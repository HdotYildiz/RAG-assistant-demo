# ADR 0001: Ground Answers Only in the FreshStack Corpus

## Status

Accepted

## Context

The assistant must answer technical LangChain questions from a specified corpus and make
its evidence independently verifiable.

## Decision

Search only `freshstack/corpus-oct-2024` with the `langchain` subset. The queries dataset,
accepted answers, nuggets, and relevance judgments are evaluation-only data. Generation
receives only the user question and retrieved corpus chunks.

## Consequences

- Citations can identify the exact retrieved corpus chunk.
- The assistant may not know newer LangChain behavior and must state when evidence is
  insufficient.
- Evaluation remains meaningful because its reference material cannot leak into answers.