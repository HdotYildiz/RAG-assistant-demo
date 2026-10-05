# ADR 0002: Use Hybrid Local Retrieval

## Status

Accepted

## Context

LangChain questions contain exact API names as well as semantic descriptions. Either
keyword-only or embedding-only search can miss useful evidence. The initial hybrid result is
a baseline, not evidence that hybrid retrieval is the best choice for this corpus.

## Decision

Use local BM25 sparse retrieval and FAISS dense retrieval over the same normalized chunks,
then fuse their rankings with reciprocal rank fusion. Compare this baseline with BM25-only
and dense-only retrieval on the development partition before selecting a final strategy.

## Consequences

- Exact identifiers and conceptual phrasing both contribute to retrieval.
- Indexing requires additional disk space and dependencies compared with a single index.
- Selection must consider quality, latency, and resource cost without tuning on the final partition.