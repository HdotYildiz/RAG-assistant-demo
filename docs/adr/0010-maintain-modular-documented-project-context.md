# ADR 0010: Maintain Modular, Documented Work with Current Project Context

## Status

Accepted

## Context

This prototype is being developed iteratively by people and LLM-based coding assistants.
Without clear boundaries and current documentation, a small RAG system can quickly accumulate
tightly coupled changes, stale instructions, and repeated discovery work.

## Decision

Keep changes modular: data loading, indexing, retrieval, generation, evaluation, and CLI
orchestration must remain independently understandable and testable. Document every
material behavior, configuration, decision, command, result, and limitation in the
appropriate repository document. Maintain root `CONTEXT.md` after material changes so a new
LLM or contributor can understand the system state, constraints, validation status, and
next work without reconstructing the project from source.

## Consequences

- New features should extend the owning module rather than couple unrelated layers.
- Tests must cover changed behavior at the narrowest practical boundary.
- ADRs record durable choices; README documents use; results reports record measured
  outcomes; future-work documents unresolved limitations.
- `CONTEXT.md` is a concise handoff document, not a replacement for source or detailed
  documentation, and must be updated whenever its summary becomes stale.