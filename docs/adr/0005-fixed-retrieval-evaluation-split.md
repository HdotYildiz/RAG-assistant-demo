# ADR 0005: Use a Fixed Development and Final Evaluation Split

## Status

Accepted

## Context

FreshStack provides one LangChain query set. Retrieval settings and future refusal behavior
need tuning, but reporting scores on the questions used for tuning would overstate quality.

## Decision

Deterministically split the query set by sorted query ID, a configured random seed, and a
configured development fraction. Use the development partition for retrieval tuning and run
the final partition only after settings are fixed. Initial automated metrics are Recall@$k$,
MRR@$k$, and nDCG@$k$ against the union of nugget-level relevant corpus IDs.

## Consequences

- Results are reproducible from configuration without storing evaluation data in the index.
- Evaluation answers and nuggets remain outside the assistant's retrieval and generation
  inputs.
- Final metrics should not be updated after parameter tuning without documenting a new split
  or evaluation run.