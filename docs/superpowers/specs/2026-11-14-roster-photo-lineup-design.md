# Roster-photo lineup assistant — design (deferred)

**Status: deferred.** Brainstormed 2026-11-14, intentionally not scheduled — Phase 7
(deployment) goes first. This doc exists so the idea isn't lost, not as a build-now spec.
Revisit after Phase 7/8 are done, or sooner if the user asks.

## Intent

Today the chat assistant only reasons about players the caller names or `@`-mentions by
hand — it has no concept of "my team." The user wants to upload a screenshot of their
actual fantasy roster (ESPN or Sleeper) and get real lineup decisions against it: both the
common case ("should I start X or Y in my last flex spot") and the less common one ("give
me my full optimal starting lineup, with reasoning").

This is a lighter-weight alternative to a full ESPN/Sleeper API league connector (already
tracked as a deferred item — see `ffw_future_league_scoring_and_connectors` memory): no
OAuth, no league ID, no live sync — just a photo, read once per session. Trade-off: no
live injury/bye-week feed, and extraction can misread a cropped or stale screenshot.

## Goals

- Upload a roster screenshot (ESPN or Sleeper) once at the start of a session; the
  extracted roster (names + starter/bench slots) persists for the rest of that
  conversation — no re-upload per question.
- Support single flex-spot-style comparisons ("X or Y in my flex") using the roster as the
  source of candidate players, reusing the existing decision engine/context builder as-is.
- Support "optimize my full lineup": compare bench players against starters **across
  positions**, not just within the same slot — e.g. flag a bench WR as better flex value
  than a started RB, not only RB-vs-RB swaps.
- Before answering anything, show the extracted roster (names + slots) back to the user
  for confirmation/correction, since vision extraction can misread a screenshot.

## Non-goals / anti-goals

- No OAuth/API integration with ESPN/Sleeper leagues — this reads a static image only.
  (Full connectors remain a separate, already-deferred item.)
- No live injury/news sync tied to the uploaded roster beyond what the existing
  stats/news pipeline already provides per player.
- Not a replacement for the existing `@`-mention flow — additive. A caller who never
  uploads a photo sees no change in behavior.
- Would be a failure if "optimize my lineup" ever silently proposed an illegal swap (wrong
  position for a slot, e.g. suggesting a TE for a pure RB spot) — slot-eligibility rules
  must be enforced, not left to the LLM's judgment.

## What's decided

- **Extraction**: Claude's vision reads the screenshot directly (no separate OCR step) and
  returns structured `(name, slot)` pairs.
- **Platforms**: ESPN and Sleeper roster screens are the two to support; their slot
  layouts/labels differ and both need their own extraction prompt/few-shot examples.
- **Session state**: the extracted roster is held for the session (exact mechanism —
  frontend-held state passed back each `/chat` call, vs. a server-side session store —
  left to the builder; either is consistent with the backend's current stateless design,
  but a server-side store is new infrastructure this app doesn't have yet).
- **Confirmation step**: always show the extracted roster back before answering any
  question against it, so the user can correct a misread name/slot first.
- **Cross-position flex logic**: when optimizing a full lineup, bench players must be
  compared against started players at *any* slot they're flex-eligible for, not just their
  own position — reuses the existing decision engine's comparison/reasoning, just run
  across a wider candidate set with slot-eligibility rules enforced.

## What's left to the builder

- Exact extraction prompt/schema per platform (ESPN vs. Sleeper screenshots differ in
  layout).
- Slot-eligibility table (which positions are flex-eligible, bench vs. IR distinction,
  etc.) — likely a small static config, not data-driven.
- Session-state mechanism (frontend state vs. a new server-side store) and its lifetime.
- UX for the confirmation/correction step (tap-to-edit, as sketched during brainstorming,
  vs. something simpler).
- Error handling for an unreadable photo, an unsupported platform, or a name extraction
  can't resolve to a known player in `data/processed/player_stats.parquet`.
