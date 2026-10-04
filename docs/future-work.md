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
