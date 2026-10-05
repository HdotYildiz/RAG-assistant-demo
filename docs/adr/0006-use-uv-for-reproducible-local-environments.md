# ADR 0009: Use uv for Reproducible Local Environments

## Status

Accepted

## Context

The system Python installation and ad hoc virtual environments produced conflicting package
availability during development. The project includes native dependencies such as FAISS and
large model libraries, making reproducible environment resolution important.

## Decision

Use `uv` as the authoritative local dependency workflow. Pin the Python version in
`.python-version`, declare dependencies in `pyproject.toml`, commit `uv.lock`, create the
repository-local environment with `uv sync`, and run commands with `uv run`.

## Consequences

- Contributors use the same resolved dependency set and Python version.
- Tests, CLI commands, and package installation target the repository-local environment
  rather than an arbitrary system interpreter.
- `uv` is a development prerequisite and lockfile updates must accompany dependency changes.