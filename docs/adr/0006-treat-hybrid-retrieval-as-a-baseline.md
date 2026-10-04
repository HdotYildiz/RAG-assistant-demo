# ADR 0006: Treat Hybrid Retrieval as a Baseline to Evaluate

## Status

Accepted

## Context

The prototype starts with BM25 plus dense FAISS retrieval fused by reciprocal-rank fusion.
The initial development result is a baseline, not evidence that hybrid retrieval is the best
choice for this corpus. It adds dependencies, disk artifacts, and runtime work compared with
either retriever alone.

## Decision

Keep hybrid retrieval as the current baseline and compare it with BM25-only and dense-only
retrieval on the fixed development partition. Select the production-oriented retrieval
strategy only after comparing Recall@$k$, MRR@$k$, nDCG@$k$, latency, and resource cost. Do
not use the held-out final partition to make that selection.

## Consequences

- The existing hybrid implementation remains useful while avoiding a premature permanent
  commitment to it.
- Evaluation must support comparable retrieval modes before final metrics are reported.
- A simpler retriever may be selected if it provides comparable quality with lower latency or
  operational cost.