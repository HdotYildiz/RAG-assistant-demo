# Future Work

This backlog captures deliberate prototype limitations and improvements that should be
addressed before relying on the assistant for sustained or shared use.

## Resumable, Atomic Index Builds

### Current Limitation

Index construction embeds the full corpus in one run and writes the final artifacts only
after all corpus loading and embedding work completes. If the process is interrupted,
embedding starts again from the beginning on the next run. An interruption during final
artifact writes can also leave a partial `.rag-index` directory, which the application
rejects as incomplete.

### Proposed Improvement

1. Embed chunks in deterministic batches and persist each completed batch with its source
   chunk IDs.
2. Record build progress, embedding-model identity, corpus version, and chunk count in a
   checkpoint manifest.
3. Resume only batches compatible with the active corpus and embedding model.
4. Build into a temporary directory and atomically rename it to `.rag-index` only after
   every required artifact is complete and validated.
5. Add a `rag-assistant index --resume` command and automated interruption/recovery tests.

### Acceptance Criteria

- An interrupted build resumes without re-embedding already checkpointed chunks.
- `ask` never reads a partial index.
- A completed index carries sufficient manifest information to detect incompatible corpus
  or embedding-model changes.

## Evidence-Based Refusal

### Current Limitation

The retriever always returns its top-ranked chunks, including for questions outside the
LangChain corpus. The assistant currently treats any nonempty result list as sufficient
evidence, so an unrelated question can receive irrelevant cited excerpts instead of a clear
refusal.

### Proposed Improvement

Use the eventual evaluation results to select an evidence gate. Compare candidate approaches
such as sparse-score thresholds, dense-similarity thresholds, fused-score thresholds, and a
small relevance classifier against the development split. The current lexical-overlap guard
is a conservative stopgap, not the final solution. Choose the option that best trades off
correct refusals against missed supported answers, then lock it before final evaluation.

### Acceptance Criteria

- Unsupported questions return an insufficient-information response with no sources.
- The selected gate and its threshold are justified by development-set results, not an
   arbitrary constant.
- The final held-out evaluation reports both supported-answer quality and refusal behavior.

## Citation and Context Quality

### Current Limitation

Generation receives bounded raw corpus chunks, but it can still produce a citation that was
not supplied in context or respond poorly to structured source formats such as notebook JSON.

### Proposed Improvement

Validate generated inline citation IDs against the selected evidence before displaying an
answer. Normalize structured source files into readable prose/code sections before indexing
or generation, then evaluate citation support against FreshStack judgments.
