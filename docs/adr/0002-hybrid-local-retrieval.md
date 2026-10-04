# ADR 0002: Use Hybrid Local Retrieval

## Status

Accepted

## Context

LangChain questions contain exact API names as well as semantic descriptions. Either
keyword-only or embedding-only search can miss useful evidence.

## Decision

Build a local BM25 sparse index and a FAISS dense index over the same normalized chunks.
Fuse ranked results using reciprocal rank fusion. Make a cross-encoder reranker optional so
the prototype can run on modest hardware.

## Consequences

- Exact identifiers and conceptual phrasing both contribute to retrieval.
- Indexing requires additional disk space and dependencies compared with a single index.
- The implementation can offer a fast mode without reranking and a quality mode with it.