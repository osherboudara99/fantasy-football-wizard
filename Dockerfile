# syntax=docker/dockerfile:1

# API image for Cloud Run (README §13). Only the FastAPI request path's own
# code and dependencies go in here: api/, pipeline/, llm/, retrieval/, and the
# base (non-extra) dependency group - no scripts/, embeddings/, frontend/,
# tests/, or data/, and no nflreadpy/pandas/sentence-transformers/torch (the
# refresh jobs' deps - see pyproject.toml's "refresh" extra). Embedding only
# ever runs in embeddings/build_embeddings.py, in the refresh job, never here.

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Deps in their own layer, cached separately from application code so a code
# change doesn't force a dependency re-resolve. --frozen: fail rather than
# silently drift from the committed uv.lock. Base group only (no extras).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.11-slim
WORKDIR /app

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /app/.venv /app/.venv
COPY api/ ./api/
COPY pipeline/ ./pipeline/
COPY llm/ ./llm/
COPY retrieval/ ./retrieval/
RUN chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

# Cloud Run injects $PORT at runtime (defaults to 8080) and expects the
# container to listen on it - shell form so the variable actually expands.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
