# Results Report

## Scope

This report records two separate development evaluations for the Technical Knowledge
Assistant:

1. A retrieval evaluation: can the system find the corpus chunks FreshStack judges relevant?
2. A generated-answer evaluation: when a local model reads retrieved chunks, does it produce
  a useful answer with inline citations and refuse when evidence is not sufficient?

The retrieval evaluation and generated-answer evaluation answer different questions. A good
retrieval result does not prove that a model will produce a grounded answer.

The detailed, machine-readable rankings are in
[results/retrieval-development.json](../results/retrieval-development.json).

## Evaluation 1: Retrieval

### Why We Did It

Before evaluating generated answers, we needed a baseline for whether the system retrieves
the relevant FreshStack corpus chunks for a question.

### What We Did

| Setting | Value |
| --- | --- |
| Corpus | `freshstack/corpus-oct-2024`, `langchain`, `train` |
| Corpus size | 49,514 chunks |
| Query dataset | `freshstack/queries-oct-2024`, `langchain`, `test` |
| Split | 70% development / 30% final, seed `42` |
| Development queries | 142 |
| Retrieval | BM25 plus FAISS dense retrieval, fused with reciprocal-rank fusion |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Candidate depth | 40 per retriever |
| Reported depth | $k=8$ |

For each query, relevance is the union of corpus chunk IDs attached to its FreshStack
nuggets. The evaluation uses those judgments only for measurement; they are never supplied
to the assistant at question-answering time.

### Results

| Metric | Result |
| --- | ---: |
| Recall@8 | 0.1525 |
| MRR@8 | 0.3797 |
| nDCG@8 | 0.1995 |

Recall@8 is the fraction of each query's judged-relevant chunks retrieved in the first eight
results, averaged across queries. MRR@8 measures the rank of the first judged-relevant
chunk. nDCG@8 rewards placing multiple judged-relevant chunks nearer the top of the ranking.

The baseline finds a relevant chunk relatively early for some questions, as reflected in its
MRR, but recovers a small share of all judged evidence at $k=8$. The result is a baseline,
not a final quality claim: retrieval parameters have not yet been tuned on the development
partition and the final partition has not been evaluated.

### Examples

### Successful Retrieval

For query `78366661`, concerning an empty MongoDB vector search result with LangChain, the
top-ranked chunk was `langchain/templates/rag-mongo/ingest.py_0_1219`. FreshStack judges
that chunk relevant. This is an example where the retriever's first result matches both the
technology and the task described by the question.

### Retrieval Miss

For query `78574282`, concerning a LangChain OpenAI API-key rate-limit error, none of the
top eight retrieved chunks was judged relevant. Its first result was
`azure-openai/End_to_end_Solutions/AOAISearchDemo/notebooks/structured_data_retrieval_nltosql.ipynb_0_7651`,
which is an Azure OpenAI notebook but does not address the judged evidence for the question.
This illustrates that keyword and semantic overlap alone can produce plausible-looking but
unsupported context.

## Evaluation 2: Generated Answers with Ollama

### Why We Did It

Retrieval metrics do not show whether the model follows the grounding instructions. This
evaluation checks whether a local Ollama model can turn retrieved chunks into useful answers,
show its sources inline, and avoid unsupported answers.

### What We Did

We selected 25 deterministic questions from the development partition and generated one
answer per question. The assistant received only the question and retrieved corpus chunks.
FreshStack nuggets and relevance judgments were used afterward to score the output; they were
not sent to the assistant.

The detailed artifact is
[results/answers-development-25.json](../results/answers-development-25.json).

### Results

| Metric | Result |
| --- | ---: |
| Questions evaluated | 25 |
| Nugget coverage | 0.3378 |
| Answers with an inline citation | 0.0400 |
| Inline citation IDs valid | 1.0000 |
| Supported-question refusal rate | 0.0000 |
| Unsupported-question refusal rate | 0.0000 |

The result is not sufficient for a grounded assistant. Only one of 25 answers contained an
inline citation, although the model was asked to cite factual claims. Nugget coverage was
also low: the answers contained only about one third of the relevant reference content under
the lexical scoring rule.

The reported citation-ID validity of 1.0000 is not a quality success. Answers without inline
citations are counted as valid because an empty list has no invalid ID. The metric therefore
shows that the citation filter worked when a citation appeared, not that the answers were
properly cited.

The sample answers also show the model saying that information is insufficient and then
continuing with generic or speculative advice. The current refusal metric only measures the
pre-generation evidence gate, so that behavior is not counted as a refusal.

### What We Propose Next

1. Require a generated answer to contain an inline citation from the retrieved evidence. If
  it does not, return the existing extractive answer with source IDs instead.
2. Treat the model saying that evidence is insufficient as a refusal, and stop it from adding
  speculative advice afterward.
3. Make the generation instruction shorter and stricter: answer only what the chunks prove;
  do not add generic troubleshooting or unverified code.
4. Record how many evaluated questions are supported and unsupported so the refusal rates
  have clear denominators.
5. Compare this generated mode with the extractive fallback and with simpler retrieval
  settings on the development partition. Do not run the final partition until a configuration
  is selected.

## Limitations and Next Steps

- The retrieval baseline and the local Ollama generated-answer evaluation are separate
  development results. Neither is a final quality claim.
- The local Ollama generated-answer configuration is not adequate for grounded use. Claim-level
  citation support remains unmeasured.
- The OpenAI-compatible generation endpoint returned HTTP 429 during manual testing, so the
  generated-answer evaluation uses local Ollama rather than a hosted model.
- The current assistant does not yet use a calibrated evidence gate. It can retrieve and
  cite irrelevant chunks for unsupported questions; selecting that gate is tracked in
  [future work](future-work.md).
- The final partition remains untouched. Use the development partition to compare a small
  number of retrieval settings, lock the selected configuration, then run the final
  partition exactly once.
- Index creation is slow and not resumable. See [future work](future-work.md) for the
  checkpointing plan.

## Reproduction

```powershell
uv sync --extra dev
uv run rag-assistant evaluate --partition development
uv run rag-assistant evaluate-answers --partition development
```

The first command writes retrieval results; the second writes a 25-question generated-answer
artifact. After choosing a fixed configuration from development results, run the held-out
retrieval evaluation once:

```powershell
uv run rag-assistant evaluate --partition final
```