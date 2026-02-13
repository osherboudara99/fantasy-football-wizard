# Repository Guidelines

## Architecture Overview
This project uses a backend-orchestrated LLM pipeline: structured NFL data is queried deterministically (Pandas/Polars), unstructured news is retrieved via RAG (Chroma), and the LLM provides explanations. Orchestration lives in Python modules rather than prompt-only logic.

## Build Source of Truth
- Use `README.md` in the repository root as the canonical implementation checklist and build plan for this app.
- When planning or implementing features, align work to the phases and constraints documented in root `README.md`.
- If other files conflict with build steps, prefer root `README.md` unless the user explicitly overrides it.

## Project Structure & Module Organization
- `scripts/` contains one-off utilities like `refresh_stats.py` that pull data and write Parquet outputs under `data/stats/`.
- `pipeline/` holds orchestration logic (currently `context_builder.py`).
- `llm/` contains LLM-facing interfaces (currently `interface.py`).
- `retrieval/` is reserved for news ingestion and vector search; see `retrieval/README.md` for the current placeholder note.
- `data/` is local data storage. Generated files live in `data/stats/{raw,staged,processed}`.
- `app/` is present but currently empty; UI work should land here when added.

## Build, Test, and Development Commands
- `uv venv --python 3.11` creates a local virtual environment in `.venv/`.
- `uv sync` installs dependencies from `pyproject.toml` into `.venv/`.
- `./scripts/setup_uv.sh` is a convenience wrapper for `uv venv` + `uv sync` on bash.
- `python scripts\refresh_stats.py` refreshes weekly and seasonal stats and writes Parquet to `data/stats/`.
- `pytest` runs the test suite (none committed yet).
- `ruff check .` runs linting (optional, dev dependency).

## Coding Style & Naming Conventions
- Python only; target `>=3.10` per `pyproject.toml`.
- Use 4-space indentation and standard PEP 8 naming: `snake_case` for functions/vars, `PascalCase` for classes.
- Keep data pipeline methods chainable (see `Stats` in `scripts/refresh_stats.py`).

## Testing Guidelines
- Framework: `pytest` (declared in dev dependencies).
- Place tests under `tests/` using `test_*.py` filenames.
- Prefer small, deterministic tests for data transforms (Polars/Pandas inputs, Parquet outputs).

## Commit & Pull Request Guidelines
- Recent commits use short, descriptive summaries (e.g., `updated packages, minor readme update`). Follow the same style.
- PRs should include: a brief summary, the commands run (e.g., `pytest`, `ruff check .`), and any new data artifacts generated.

## Configuration & Data Notes
- Local configuration should go in `.env` (already present). Do not commit secrets.
- Generated data in `data/` is local state; avoid committing large artifacts unless explicitly intended.
