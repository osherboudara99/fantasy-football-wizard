# Fantasy Football Wizard — Build-Out Goal Prompt

## Mission
Advance C:\dev\Projects\Fantasy_Football_Wizard from its current state (Phases 0-1
done) through Phases 2-8 of the README.md roadmap, one phase at a time, in order.
Do not skip ahead, do not invent scope beyond the README's stated non-goals
(draft strategy, trades, waiver optimization, DFS/betting, full-season sims).

Source of truth: README.md → "Development Roadmap" table. If any other doc
conflicts with README.md, README.md wins unless the user overrides explicitly.
Conventions: CLAUDE.md (PEP 8, raw→staged→processed Parquet staging, Haiku-4.5
default LLM, never reintroduce openai/llama-cpp-python).

## NEVER DO WITHOUT EXPLICIT USER APPROVAL
This overrides everything else below. Before any of the following, stop and ask
the user — do not proceed on your own judgment, even if it seems obviously safe:
- `git push`, `git push --force`, force-pushing anything, especially to main
- `git reset --hard`, `git clean -f`/`-fd`, `git checkout --`/`git restore` that
  would discard uncommitted work, `git branch -D`, deleting any branch
- Amending or rebasing published/pushed commits
- Deleting files or directories outside of files you created earlier in the
  same run (scratch/temp output is yours to clean up freely; the user's
  existing files, `data/`, `.env`, or anything under version control is not)
- Dropping/overwriting anything in `data/` that isn't trivially regenerable
  from a fresh `python scripts/refresh_stats.py` run
- Opening or merging a PR, closing/commenting on an issue, or any action
  visible to others outside the local repo
- Installing, upgrading, or removing any dependency not already pinned in
  `pyproject.toml`
- Any Phase 7 deployment step: creating/modifying GCP resources, Cloud Run
  services, GCS buckets, Cloud Scheduler jobs, Cloudflare Pages projects, or
  anything that touches a real cloud account or could incur billing
- Signing up for, or calling, any paid API (the §1.5 "deliberately excluded"
  list — Fantasy Nerds, SportsDataIO, MySportsFeeds — stays excluded)
- Skipping git hooks (`--no-verify`) or bypassing signing
If uncertain whether an action is destructive or externally visible, treat it
as if it is and ask.

## Checkable goals (one phase = one unit of work)

| Phase | Done-check (verbatim from README) | How to verify |
|---|---|---|
| 2. Context builder | `build_context(["Player A","Player B"], week)` returns the §7 comparison format; unit-tested with fixture data | `pytest tests/test_context_builder.py` green, manual call in a REPL matches §7's example shape |
| 3. Decision engine | A real two-player question answers correctly from the terminal | Run the CLI/script end-to-end against real data for a known week; sanity-check the `Recommendation` fields make sense |
| 4. FastAPI backend | `curl` returns a recommendation JSON | `uvicorn api.main:app --reload` then `curl -X POST localhost:8000/recommendation -d '{...}'` returns 200 + valid schema |
| 5. React frontend | Full flow works locally against FastAPI | Manually drive the UI in a browser (per the `run` skill) — player pick → question → recommendation renders |
| 6. News RAG | Recommendations cite recent news | Context builder output includes ≥1 news snippet dated within 7 days for a player with active news |
| 7. Deployment | Public URL serves a recommendation | `curl` against the deployed Cloud Run URL returns a real recommendation |
| 8. Evaluation | Week-over-week agreement tracking exists | A log/report exists comparing at least one week's recommendations to FantasyPros consensus |

A phase is **not done** until its done-check passes AND `README.md`'s roadmap
table row AND `CLAUDE.md`'s "Development Roadmap (status)" section are both
updated to say so (CLAUDE.md already mandates this).

## Per-phase loop
1. Read the phase's README section(s) referenced in the "Refs" column before writing code.
2. Implement the smallest slice that satisfies the done-check — no gold-plating, no unrequested abstractions (see CLAUDE.md "Doing tasks").
3. Write/run tests (`pytest`) and lint (`ruff check .`) — both must be clean.
4. Run the **separate checker** (below) — do not self-certify.
5. Update `README.md` roadmap table + `CLAUDE.md` status line for that phase.
6. Append one entry to `PROGRESS_LOG.md` (create it if it doesn't exist — see Logging).
7. Commit — one logical commit per fix/feature, matching the split-commit style already established (see git log on `feat/structured-data-ingestion`). No co-authored-by tag. Committing locally is fine without asking; pushing is not (see guardrails above).
8. Check the hard-stop gates before starting the next phase.

## Tools
- Read/Edit/Write/Glob/Grep for all code changes.
- Bash/PowerShell for `uv sync`, `pytest`, `ruff check .`, and non-destructive `git` (status/diff/log/add/commit).
- `gh` CLI is installed and authenticated (`osherboudara99`) — read-only use only (`gh pr view`, `gh api .../comments`) unless the user asks for a PR to be opened.
- WebFetch only for library/API docs (nflreadpy, FastAPI, Chroma reference pages) — never to guess undocumented endpoints.
- No new dependencies outside `pyproject.toml`'s existing stack without flagging it to the user first.

## Memory & continuity
- `README.md` roadmap table is the durable phase-completion record — always update it, never let it drift from reality.
- `PROGRESS_LOG.md` (new, append-only) is the run-by-run activity log — see Logging below.
- Claude Code auto-memory (`MEMORY.md` + linked files) holds *decisions and constraints* (stack choices, vetted data sources, user feedback) — consult `stack-decisions.md` and `data-sources-vetted.md` before introducing any new tool or data source; update them if a new durable decision gets made mid-build.
- Don't duplicate roadmap/progress tracking into memory — memory is for the *why*, the log/README are for the *what/when*.

## Separate checker (independent verification)
After step 3 (tests+lint green) and before marking a phase done, spawn a **fresh** subagent with no memory of the implementation session to:
1. Re-run the literal done-check from the table above itself, from scratch.
2. Run a `code-review` pass (medium effort minimum) on the diff for correctness bugs, specifically watching for silent mismatched joins/timeframes — the class of bug already found once in this repo (the Codex findings on `refresh_stats.py`'s season/week resolution and ECR mismatch).
Only mark the phase complete if the checker independently confirms both. If it disagrees with the builder, the builder's self-report loses — fix the issue, don't argue the checker down.

## Logging
- `PROGRESS_LOG.md` entry per phase: date, phase number, one-line summary of what shipped, done-check result (pass/fail + how verified), test/lint status, approximate LLM $ spent this phase.
- Keep using the existing `log()` helper pattern (prefixed `[module_name] msg`) for any new script's runtime logging (`refresh_news.py`, `build_embeddings.py`, etc.) — consistent with `refresh_stats.py`.
- Commit messages stay short and describe the *why*, matching this session's style.

## Cost sense
- Default LLM is `claude-haiku-4-5` everywhere (per README §9) — only use `ANTHROPIC_MODEL=claude-sonnet-5` where the README explicitly calls for harder reasoning (Phase 3), and say so out loud when switching.
- Don't refetch nflreadpy/Sleeper raw data that's already on disk for the current day — reuse `data/raw/*.parquet` within a session instead of re-hitting the network each iteration.
- No paid services, ever (see NEVER DO section above).
- Log approximate $ per phase in `PROGRESS_LOG.md`; a phase burning more than ~$1 in LLM calls during dev is a signal something's looping wastefully — investigate before continuing.
- Embeddings (Phase 6) run `sentence-transformers` locally, not via a paid embedding API — no torch in the API image (README §13).

## Hard stop
Never cross a phase boundary without: tests green + lint clean + checker sign-off + roadmap/CLAUDE.md updated + commit made. Beyond the NEVER DO list above, also stop and ask the user if:
- A single autonomous run exceeds roughly 2 hours of wall-clock work
- 3 consecutive attempts at the same done-check fail — surface the blocker with what was tried and stop, rather than looping

## Definition of done
Phases 2-8 done-checks all pass, README + CLAUDE.md both read "done" for every
phase, `PROGRESS_LOG.md` has one entry per phase, and a final summary is given
to the user. Then stop — do not start inventing v2 features.
