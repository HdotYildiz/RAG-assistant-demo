"""Command-line entry point for the technical knowledge assistant."""

import typer

from technical_knowledge_assistant.config import settings
from technical_knowledge_assistant.data import load_corpus
from technical_knowledge_assistant.generation import GroundedAnswerGenerator
from technical_knowledge_assistant.indexing import HybridIndex, build_index
from technical_knowledge_assistant.retrieval import HybridRetriever

app = typer.Typer(no_args_is_help=True, help="Grounded LangChain knowledge assistant.")


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
    corpus_index = HybridIndex.load(settings.index_directory)
    evidence = HybridRetriever(corpus_index).search(
        question,
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
    )
    answer = GroundedAnswerGenerator(settings).answer(question, evidence)
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nSources:")
        for citation in answer.citations:
            typer.echo(f"- {citation}")
    elif not answer.sufficient_evidence:
        typer.echo("\nSources: none")