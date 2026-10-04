"""Command-line entry point for the technical knowledge assistant."""

import typer

from technical_knowledge_assistant.config import settings
from technical_knowledge_assistant.data import load_corpus
from technical_knowledge_assistant.evaluation import evaluate_retrieval, load_queries, save_result
from technical_knowledge_assistant.evaluation.freshstack import split_queries
from technical_knowledge_assistant.generation import GroundedAnswerGenerator
from technical_knowledge_assistant.indexing import HybridIndex, build_index
from technical_knowledge_assistant.models import Answer
from technical_knowledge_assistant.retrieval import HybridRetriever

app = typer.Typer(no_args_is_help=True, help="Grounded LangChain knowledge assistant.")


def _load_assistant() -> tuple[HybridRetriever, GroundedAnswerGenerator]:
    """Load the persisted index and initialize the per-process answer components."""
    corpus_index = HybridIndex.load(settings.index_directory)
    return HybridRetriever(corpus_index), GroundedAnswerGenerator(settings)


def _answer_question(
    question: str,
    retriever: HybridRetriever,
    generator: GroundedAnswerGenerator,
) -> Answer:
    """Retrieve evidence and produce one grounded answer."""
    evidence = retriever.search(
        question,
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
    )
    return generator.answer(question, evidence)


def _display_answer(answer: Answer) -> None:
    """Render a grounded answer and its source chunk IDs consistently."""
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nSources:")
        for citation in answer.citations:
            typer.echo(f"- {citation}")
    elif not answer.sufficient_evidence:
        typer.echo("\nSources: none")


@app.command()
def status() -> None:
    """Show local runtime configuration without exposing credentials."""
    typer.echo(f"Embedding model: {settings.embedding_model}")
    typer.echo(f"Index directory: {settings.index_directory}")
    typer.echo(f"LLM provider: {settings.llm_provider}")
    typer.echo(f"LLM model: {settings.llm_model or 'not configured'}")


@app.command()
def index() -> None:
    """Build and save the local searchable corpus index."""
    typer.echo(f"Loading {settings.corpus_dataset} ({settings.corpus_subset})...")
    chunks = load_corpus(settings.corpus_dataset, settings.corpus_subset, settings.corpus_split)
    typer.echo(f"Embedding {len(chunks)} corpus chunks with {settings.embedding_model}...")
    corpus_index = build_index(chunks, settings.embedding_model)
    corpus_index.save(settings.index_directory)
    typer.echo(f"Saved hybrid index to {settings.index_directory}.")


@app.command()
def ask(question: str) -> None:
    """Answer one question using only evidence from the local corpus index."""
    retriever, generator = _load_assistant()
    _display_answer(_answer_question(question, retriever, generator))


@app.command()
def chat() -> None:
    """Ask multiple questions while keeping the index and embedding model loaded."""
    retriever, generator = _load_assistant()
    typer.echo("Technical Knowledge Assistant. Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            question = typer.prompt("Question").strip()
        except typer.Abort:
            typer.echo("\nSession ended.")
            return
        if question.lower() in {"exit", "quit"}:
            typer.echo("Session ended.")
            return
        if not question:
            continue
        _display_answer(_answer_question(question, retriever, generator))


@app.command()
def evaluate(partition: str = typer.Option("development", help="Evaluation partition: development or final.")) -> None:
    """Measure retrieval quality against held-out FreshStack relevance judgments."""
    if partition not in {"development", "final"}:
        raise typer.BadParameter("Partition must be 'development' or 'final'.")
    typer.echo(f"Loading {settings.query_dataset} ({settings.query_subset}) evaluation queries...")
    queries = load_queries(settings.query_dataset, settings.query_subset, settings.query_split)
    development_queries, final_queries = split_queries(
        queries,
        development_fraction=settings.development_fraction,
        seed=settings.evaluation_seed,
    )
    selected_queries = development_queries if partition == "development" else final_queries
    retriever, _ = _load_assistant()
    typer.echo(f"Evaluating {len(selected_queries)} {partition} queries...")
    result = evaluate_retrieval(
        retriever,
        selected_queries,
        candidates=settings.retrieval_candidates,
        k=settings.retrieval_top_k,
        partition=partition,
        progress_callback=lambda completed, total: typer.echo(f"  Completed {completed}/{total} queries."),
    )
    output_path = save_result(result, settings.evaluation_directory)
    metrics = result.metrics
    typer.echo(f"Evaluated {metrics.query_count} queries at k={result.k}.")
    typer.echo(f"Recall@{result.k}: {metrics.recall_at_k:.4f}")
    typer.echo(f"MRR@{result.k}: {metrics.mrr_at_k:.4f}")
    typer.echo(f"nDCG@{result.k}: {metrics.ndcg_at_k:.4f}")
    typer.echo(f"Saved results to {output_path}.")