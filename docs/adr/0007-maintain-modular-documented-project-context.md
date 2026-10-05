# ADR 0010: Keep Components Modular and Swappable

## Status

Accepted

## Context

Retrieval, generation, indexing, and evaluation will evolve independently. Tightly coupling
them makes it difficult to compare alternatives or reuse a component in another workflow.

## Decision

Keep data loading, indexing, retrieval, generation, evaluation, and CLI orchestration behind
clear module boundaries. Components must be independently testable and designed so an
implementation can be replaced or reused without changing unrelated layers.

## Consequences

- New features should extend the owning module rather than couple unrelated layers.
- Tests should cover behavior at the narrowest practical boundary.
- Alternative retrievers or generation providers can be evaluated without rewriting the CLI
  or evaluation workflow.