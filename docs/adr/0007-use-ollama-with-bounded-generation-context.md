# ADR 0007: Use Ollama for Local Development with Bounded Generation Context

## Status

Accepted

## Context

Hosted generation providers require credentials, credits, and rate-limit handling. The
prototype needs a usable local development path. Local models also make large raw source
contexts more expensive and more prone to poor responses, particularly when corpus chunks
contain notebook JSON or long code examples.

## Decision

Use Ollama as the preferred local generation provider through its OpenAI-compatible Chat
Completions endpoint. Keep provider selection configurable so hosted OpenAI-compatible
providers remain supported. Bound generation input and output: cap total context, cap each
chunk, and cap generated tokens. Current defaults are 12,000 context characters, 4,000
characters per chunk, and 512 generated tokens.

## Consequences

- Local development does not require API keys, paid credits, or external model calls.
- Bounded context improves local latency and prevents a few large chunks from dominating the
  prompt.
- The caps can omit supporting detail and must be evaluated before being treated as final
  production settings.
- Ollama must be installed locally and the selected model must be pulled before generation.