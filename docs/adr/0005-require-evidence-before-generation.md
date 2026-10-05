# ADR 0008: Require Evidence Before Generation

## Status

Accepted (provisional gate)

## Context

Ranked retrieval returns its nearest chunks even for questions outside the LangChain corpus.
Passing those chunks to a generation model can produce plausible but unsupported answers and
wastes local or hosted generation capacity.

## Decision

Run an evidence gate before extractive or model-based generation. The current conservative
gate requires a retrieved chunk to share at least two meaningful non-stopword terms with the
question. Only chunks passing the gate are included in the response context and sources. If
none pass, return an insufficient-information response with no sources and do not call the
generation provider.

## Consequences

- Unsupported questions such as general-knowledge prompts are declined instead of being
  answered from loosely related LangChain examples.
- The lexical gate can reject valid paraphrases and is not a final relevance model.
- Development-set experiments must calibrate or replace this provisional guard before final
  evaluation; the threshold must not be tuned on the held-out final partition.