# ADR 0003: Keep Answer Generation Provider-Pluggable

## Status

Accepted

## Context

No model API credits are supplied, and users may prefer a local model or an existing hosted
provider account.

## Decision

Define a small answer-generation interface and select its implementation from environment
configuration. Retrieval and evaluation must work without an LLM provider; answer-quality
evaluation may use deterministic nugget coverage where a judge model is unavailable.

## Consequences

- The project can be reproduced without committing credentials.
- Providers need small adapters, but core retrieval stays stable.
- Result reports must state the provider/model and any generation-evaluation method used.