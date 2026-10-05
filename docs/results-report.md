# Results Report

## Scope

This report records the first retrieval-only baseline for the Technical Knowledge
Assistant. It evaluates the hybrid BM25 plus dense-retrieval system against FreshStack
LangChain relevance judgments. It does not evaluate generated answer quality because the
initial run was retrieval-only; answer generation was not part of that evaluation run.

The detailed, machine-readable rankings are in
[results/retrieval-development.json](../results/retrieval-development.json).

## Configuration

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

## Development Baseline

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

## Examples

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

## Generated-Answer Evaluation

Generated-answer evaluation is implemented but has not yet been run for a documented
development subset. With Ollama configured, run:

```powershell
uv run rag-assistant evaluate-answers --partition development
```

The command defaults to 25 deterministically selected development queries and writes a
separate answer artifact. It reports:

- Nugget coverage: the share of FreshStack nuggets where at least 50% of meaningful nugget
  terms appear in the answer.
- Citation presence: the share of non-refused answers containing at least one inline chunk
  ID.
- Citation validity: the share of non-refused answers whose inline IDs are all selected
  evidence IDs.
- Supported-question refusal rate and unsupported-question refusal rate, based on whether
  FreshStack supplies judged-relevant corpus chunks.

These are transparent lexical and identity checks. They do not establish that a cited source
supports each individual claim, and their thresholds must be tuned only on development data.

## Limitations and Next Steps

- This report currently contains measured retrieval results only. Generated-answer metrics
  are implemented but not yet measured for a development subset; claim-level citation
  support remains unmeasured.
- Ollama local generation is available. A hosted OpenAI-compatible endpoint returned HTTP
  429 during manual testing because of limit constraints, so it was not used for this baseline.
- The assistant now has a conservative lexical evidence gate and declines questions with no
  selected evidence. Its threshold has not been calibrated on the development partition and
  may reject valid paraphrases; calibration remains [future work](future-work.md).
- Model-emitted inline citation IDs are removed unless they were present in the selected
  evidence context. This verifies citation identity, not whether a cited chunk supports a
  specific claim.
- The final partition remains untouched. Use the development partition to compare a small
  number of retrieval settings, lock the selected configuration, then run the final
  partition exactly once.
- Index creation is slow and not resumable. See [future work](future-work.md) for the
  checkpointing plan.

## Reproduction

```powershell
uv sync --extra dev
uv run rag-assistant evaluate --partition development
```

The command writes the result file used by this report. After choosing a fixed configuration,
run the held-out evaluation with:

```powershell
uv run rag-assistant evaluate --partition final
```