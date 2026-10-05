# ADR 0004: Persist Local Index Artifacts

## Status

Accepted

## Context

The corpus and embedding model download are expensive enough that rebuilding on each CLI
question would make the prototype slow and hard to reproduce.

## Decision

Persist four local artifacts in the configured index directory: normalized chunks in JSONL,
the BM25 index as a Python pickle, the FAISS index, and a JSON manifest containing the
embedding model and chunk count.

## Consequences

- `ask` starts from an existing local index instead of reloading the dataset.
- Citation IDs remain mapped to their original normalized corpus records.
- BM25 pickle files are trusted local artifacts and must not be loaded from untrusted
  sources.