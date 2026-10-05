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

### Run 1: Initial Generated Answers

The first generated-answer run is recorded in
[results/answers-development-25_v1.json](../results/answers-development-25_v1.json).

### Results

| Metric | Result |
| --- | ---: |
| Questions evaluated | 25 |
| Nugget coverage | 0.3378 |
| Answers with an inline citation | 0.0400 |
| Inline citation IDs valid | 1.0000 |
| Supported-question refusal rate | 0.0000 |
| Unsupported-question refusal rate | 0.0000 |

This result was not sufficient for a grounded assistant. Only one of 25 answers contained an
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

### Run 2: Citation Contract and Compact Fallback

After the first run, the assistant was changed to require `[[chunk-id]]` citations, refuse
when the model says evidence is insufficient, and fall back to two compact cited excerpts
when a generated answer does not contain a valid citation. The double-bracket format avoids
mistaking brackets inside code or Markdown for citations.

The second run used the same 25 deterministic development questions. Its artifact is
[results/answers-development-25_v2.json](../results/answers-development-25_v2.json).

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Nugget coverage | 0.3378 | 0.0946 |
| Answers with an inline citation | 0.0400 | 1.0000 |
| Inline citation IDs valid | 1.0000* | 1.0000 |
| Supported-question refusal rate | 0.0000 | 0.4400 |
| Unsupported-question refusal rate | 0.0000 | 0.0000** |

\* In run 1, this result was misleading because answers with no citation were counted as valid.

\** The selected 25-question subset contains no FreshStack questions without judged-relevant
chunks, so this rate has no denominator and does not measure refusal quality for unsupported
questions.

The citation result is now meaningful: delivered non-refused answers use the explicit
double-bracket format and all displayed IDs came from selected evidence. This is a real
improvement in source identity and presentation.

The answer-quality result is worse. Nugget coverage fell to 0.0946 and 44% of supported
questions were refused. The stricter contract stopped unsupported-looking synthesis, but the
local model often chose refusal even when FreshStack judged relevant evidence to exist.
Compact fallback excerpts are safer than the earlier long raw chunks, but they are still only
evidence display, not a useful synthesized answer. This configuration is therefore not ready
for a final evaluation or for normal use as a technical assistant.

### Ollama Limitations and Possible Improvements

The observed behavior is consistent with limitations of the tested local Ollama model and the
current evidence pipeline:

- The model does not reliably follow a sentence-level citation contract when the context is
  long, mixed-format, or only weakly relevant.
- It cannot determine whether a selected chunk truly supports a generated claim. A valid ID
  proves only that the chunk was supplied, not that the claim is entailed by it.
- The corpus contains raw source code and notebook JSON. These are difficult for a small local
  model to turn into a concise answer and can cause it to refuse despite relevant material.
- Retrieval remains the limiting factor. The retrieval baseline misses judged evidence for
  some questions, and no generator can recover information that was not retrieved.

Potential improvements, all to be selected using development data, are:

1. Compare BM25-only, dense-only, and hybrid retrieval, then tune candidate depth, top-$k$,
   and the evidence gate before changing generation further.
2. Normalize notebook and source-code chunks into readable prose and focused code sections
   before indexing, so the model receives smaller and clearer evidence.
3. Test a stronger local Ollama model that fits the available hardware, while keeping the same
   prompt and development subset for a fair comparison.
4. Measure model-generated citation compliance separately from extractive-fallback citations,
   and add a small manual claim-to-source review before claiming citation support.
5. Add evaluation questions that are known to be unsupported if refusal behavior must be
   measured; the current FreshStack subset does not provide them in this sample.

### Next Steps

Use the development partition to improve retrieval and evidence preparation first. Keep the
citation contract and refusal safeguards, but do not run the final partition until a
configuration improves supported-answer coverage without reintroducing speculative answers.

## Limitations and Next Steps

- The retrieval baseline and the local Ollama generated-answer evaluation are separate
  development results. Neither is a final quality claim.
- The second local Ollama run has reliable citation identity, but its low nugget coverage and
  high supported-question refusal rate make it inadequate for grounded use. Claim-level
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