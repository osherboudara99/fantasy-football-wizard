# Comprehensive chat assistant — design

**Status**: approved for implementation
**Date**: 2026-09-09

## Summary

Replace the current single-purpose start/sit form with a chat interface that
answers any fantasy-football question the existing data can support: start/sit,
trade evaluation, and general stat/prediction questions about mentioned
players. Answers cite the news sources they drew on. Players are referenced via
`@`-mention autocomplete instead of dropdown pickers or free-text name typing.

This supersedes CLAUDE.md's Project Goal line ("a decision-support system, not
a chatbot") and removes trade evaluation from the non-goals list. **Draft
strategy/ADP/rankings remain explicitly out of scope** — deferred to a later
task, not part of this change. CLAUDE.md will be updated to reflect both.

## Why

The current tool only answers one question shape (compare exactly two named
players for start/sit) through a form UI. The user wants one general-purpose
conversational entry point that reuses the existing structured-data +
news-RAG pipeline across more question types, without adding new data sources
or infrastructure this round.

## Architecture

### Today

- `pipeline/decision_engine.decide(question, players, week)` → rigid
  `Recommendation` (start/bench/confidence/key_factors/risk_factors), built via
  `context_builder.build_context()` (structured stats + `news_fn`) then
  `llm.interface.run_llm()` with `output_format=Recommendation`.
- `api/main.py`: `POST /recommendation` (exactly 2 players), `GET /players`,
  `GET /health`.
- Frontend: two player-picker `<select>`s + submit, renders the fixed
  recommendation shape.

### New

One generalized pipeline replaces the rigid two-player schema, since not every
question is a binary start/bench call.

**`POST /chat`** (replaces `POST /recommendation`):
```
Request:
{
  "message": str,                    # free-text question
  "mentioned_players": [str],        # exact names from the @-mention autocomplete (GET /players), may be []
  "week": int | null,                # defaults to target_week()
  "history": [{"role": "user"|"assistant", "content": str}]  # prior turns, frontend-held
}

Response:
{
  "answer": str,                     # free-form prose answer
  "sources": [{"title": str, "link": str, "source": str, "published_at": str}],
  "players_discussed": [str],        # resolved player names, for the UI to label/highlight
  "recommendation": {                # present only when the question is start/sit-shaped
    "start": str, "bench": str, "confidence": float,
    "key_factors": [str], "risk_factors": [str]
  } | null
}
```

**Player resolution** (`pipeline/chat_engine.py`, new): `mentioned_players`
are exact names already validated by the `@`-mention autocomplete (sourced
from `GET /players`, the same `known_player_names()` list used everywhere
else) — no player_id needed at the API boundary, consistent with how
`RecommendationRequest.players`/`extract_players` already treat names as the
canonical identifier. Any player named in free text but *not* `@`-mentioned
still goes through the existing `pipeline.entity_extraction.extract_players`
as a fallback, so typing a name without mentioning it still works. The
combined, de-duplicated (case-insensitive) player list can be any length ≥ 1
(today's `REQUIRED_PLAYERS == 2` constraint is dropped for chat — start/sit-
style questions still expect 2, but a "how many points will X score"
question expects 1, and a 3-way flex question is now representable too).

**Context building**: `pipeline.context_builder.build_context()` already loops
over an arbitrary `players: list[str]` — no change needed there beyond
widening the news-source contract (below).

**News sources**: `retrieval.news_retriever.retrieve_news()` currently returns
`list[str]` (title+description concatenated, metadata discarded). Sources
require `link`/`source`/`published_at`, so `retrieve_news` changes its return
type to `list[NewsItem]` where `NewsItem` is a small dataclass/TypedDict
(`title`, `snippet`, `link`, `source`, `published_at`). `context_builder`
formats `NewsItem.snippet` into the existing "Recent news" bullet (prompt text
unchanged); `chat_engine` separately collects every `NewsItem` used across all
mentioned players into the response's `sources` list, de-duplicated by
`link`.

**LLM call**: `llm/interface.py` gains a second structured-output schema,
`ChatAnswer` (`answer: str`, `recommendation: Recommendation | None`), used by
a new `run_chat_llm(context, message, history)` alongside the existing
`run_llm`/`Recommendation` (kept for the CLI entrypoint —
`python -m pipeline.decision_engine` stays a direct two-player tool; see
Non-goals). `history` is passed as prior `messages` entries to
`client.messages.parse` so multi-turn context reaches Claude without any
server-side storage. The system prompt is extended to tell Claude: (a) only
use the provided context, never outside knowledge of specific players, (b)
populate `recommendation` only for genuine start/sit-shaped questions, else
leave it `null`, (c) reasoning for trade questions should weigh both sides'
provided stats/projections/news directly — no separate trade-value model.

**Trade evaluation**: no new data source. Both traded players'/sides' stats
and news go through the same `build_context()` call; the LLM reasons over the
combined context. Explicitly not building a formal value/ranking model this
round.

**`/recommendation` and `decide()`**: `POST /recommendation` is removed from
`api/main.py` once `/chat` is live. `pipeline/decision_engine.decide()` and its
CLI stay as-is (still useful standalone, and `tests/test_decision_engine.py`
keeps covering it) — `chat_engine` is a new, separate orchestration module
that does not call `decide()`, since `decide()`'s player-count and schema
assumptions don't generalize. Some logic (`resolve_week`, `target_week`) is
shared by importing from `decision_engine` rather than duplicating.

### `@`-mention player search

Frontend-only addition: typing `@` in the chat textarea opens an autocomplete
dropdown filtered client-side against `GET /players` (already returns the
full name list; fetched once on load and cached in memory — the list is small
enough that no server-side search endpoint is needed). Selecting an entry
inserts a visible token in the message text and adds the player's exact name
to `mentioned_players` sent with the request.

### Frontend

Replace `frontend/`'s picker form with a chat view:
- Scrollable message list, user/assistant bubbles.
- Assistant bubbles render `answer` as prose, a collapsible "Sources" list
  (`title` linking to `link`, with `source` and a relative published date),
  and — only when `recommendation` is non-null — the existing decision card
  (start/bench badges, confidence bar, key/risk factors), reusing that
  component as-is.
- A small week selector stays above the input (defaults to
  `target_week()`-equivalent, overridable).
- Input box: textarea with `@`-mention autocomplete; submit sends `message` +
  current `mentioned_players` + `history` (the full prior transcript held
  in React state) + `week`.
- No persistence across page reloads (frontend-held history only, per the
  approved design decisions) — refreshing starts a new conversation.
- Visual pass via the `frontend-design` skill.

## Error handling

Same taxonomy as today, mapped to chat-bubble errors instead of hard HTTP
failures reaching the UI as broken states:
- `PlayerNotFoundError` (mentioned/extracted player has no row for the
  target week) → 404 → frontend renders an assistant error bubble ("I
  couldn't find stats for X in week N").
- `DataUnavailableError` → 503 → generic "data's unavailable, try again
  later" bubble.
- `DecisionError`/LLM output validation failure → 400 → "couldn't produce an
  answer" bubble asking the user to rephrase.
- Zero resolved players (no mention, no extractable name) → 400 from the API
  before ever calling the LLM, asking the user to `@`-mention a player.

## Testing

- `tests/test_api.py`: replace `/recommendation` tests with `/chat` tests
  (stubbed `chat_engine`), covering: mention-only resolution, text-fallback
  resolution, zero-player 400, each error mapping, `recommendation` present
  vs. `null`.
- New `tests/test_chat_engine.py` mirroring `test_decision_engine.py`'s style:
  stubbed LLM, fixture tables, asserts on player-resolution precedence
  (mention over extraction) and sources de-duplication.
- `tests/test_news_retriever.py` (existing, if present) / new coverage for
  `retrieve_news`'s changed return shape.
- Frontend: manual verification via dev server (no frontend test suite exists
  today — consistent with current project conventions).

## Non-goals (this round)

- **Draft strategy / ADP / consensus rankings / tiers** — needs new data
  ingestion (`nflreadpy.load_ff_rankings()` or similar); deferred to a later
  task. CLAUDE.md's roadmap will note this explicitly so it isn't lost.
- Streaming responses (SSE/websocket) — non-streaming request/response only.
- Backend-persisted chat history / sessions / user accounts — frontend-held
  history only.
- Waiver-wire optimization, DFS/betting advice — still permanently out of
  scope per the original non-goals.
- A dedicated trade-value/ranking model — the LLM reasons over existing
  context directly.

## Rollout

Implement this fully (backend + frontend), verify locally, then resume Phase 7
deployment (rate limiting + docs lockdown now scoped to `/chat` instead of
`/recommendation`, GCS-backed data loading, Cloud Run, Cloudflare Pages) —
each deployment step still needs its own explicit approval per
AGENT_GOALS.md.
