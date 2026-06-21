# Drill Library + Deterministic Coaching — Design Spec

**Date:** 2026-06-21
**Status:** Approved (brainstorm) — pending implementation plan
**Applies to:** the `tennis_cv` CV pipeline (`cv-pipeline/`), specifically the coaching layer.
**Related:** `prd-tennis-ai-coach.md` (canonical product), `docs/superpowers/specs/2026-06-18-courtiq-design.md` (Phase 0 design; see its "Coaching-model decision" note), `docs/superpowers/plans/2026-06-21-cv-analysis-pipeline.md` (the built pipeline).

## Problem

Today's coaching step hands the measured serve metrics to an LLM and lets it
**generate** the issues and drills freely. For a coaching app, a confident-but-wrong
drill is the worst failure mode — it can entrench a bad habit or suggest something
injury-risky — and a bigger model does not reliably prevent this. The fix is not a
smarter model but **grounding**: make drill correctness a property of vetted data and
deterministic code, and demote the LLM to a tone pass that cannot change facts.

This also addresses the existing "drill accuracy" risk flagged in the coaching-model
decision, and is independent of which LLM provider is configured.

## Goals

- Drill **content** is correct by construction (vetted library), not by model luck.
- Drill **selection** is deterministic (code maps measured faults → vetted drills).
- The LLM can only improve tone, never alter which issues/drills appear or invent facts.
- Coaching still works with the LLM disabled or unreachable (graceful degradation).
- A human (you / a coach) can read representative coaching output and sign off the library.

## Non-Goals (this version)

- Strokes other than the serve (the library covers the three serve metrics only; more
  strokes = more library entries later, no code change).
- Calibrating the metric target ranges — still placeholder constants (tracked separately;
  a wrong threshold mis-states a fault regardless of this work).
- Cross-model A/B harness (low value now that the LLM only does voice; explicitly deferred).
- The conversational follow-up coach (PRD US-004, a separate Phase 1 feature).

## Decisions (locked in brainstorming)

| Decision | Choice |
| --- | --- |
| Drill source | I draft a v1 from established serve fundamentals; flagged for human review |
| Drill selection | **Deterministic** — code maps below-target metrics → vetted drills |
| LLM role | **Optional "voice" pass** over a deterministic baseline; cannot change facts |
| Verification | Lean — unit tests + fidelity guard + a preview CLI |

## Architecture

The coaching layer becomes three well-bounded units plus a preview tool. The
`tennis_cv.pipeline.analyze` call site is unchanged — it still calls
`generate_coaching(stroke_type, score.factors, score, client=..., model=...)`.

```
score (factors with sub_scores)
        │
        ▼
assemble_coaching()  ── deterministic ──▶ Coaching baseline  (source of truth)
        │                                        │
        │                                 apply_voice() (optional, if client/LLM available)
        │                                        │
        │                                 is_faithful(baseline, rewrite)?
        │                                  ┌─────┴─────┐
        │                               yes│           │no / error / no client
        ▼                                  ▼           ▼
   (no LLM path)                      rewrite      baseline
        └──────────────── generate_coaching() returns the chosen Coaching ─────────┘
```

### 1. Drill library — `cv-pipeline/src/tennis_cv/drills.py`

Vetted data + lookup. One entry per serve fault, keyed by the metric name that, when
**below its target range**, signals that fault.

- `class Drill(BaseModel)` — `name: str`, `description: str` (moved here; re-exported from
  `coaching` for back-compat with existing imports). A concrete exercise.
- `class FaultEntry(BaseModel)` — the vetted content for "this metric is below target":
  - `metric: str` (e.g. `"knee_flexion"`)
  - `title: str` (issue headline, e.g. `"Shallow leg load"`)
  - `why: str` (why it matters, grounded in serve mechanics)
  - `how: str` (the corrective cue / how it should feel)
  - `drills: list[Drill]` (1–2 vetted drills)
- `DRILL_LIBRARY: dict[str, FaultEntry]` — entries for `knee_flexion`, `contact_height`,
  `arm_extension`. Drafted from serve canon; a top-of-file comment marks it
  **"v1 draft — pending coach review."**
- `def fault_entry(metric: str) -> FaultEntry | None` — lookup helper.

### 2. Deterministic assembly — `coaching.py`

- Constant `ISSUE_SUBSCORE_CUTOFF = 80.0` — a metric is a fault when its `sub_score`
  is **below** this; at/above it is a strength.
- `class Issue(BaseModel)` — `title, why, how, priority: int`, plus `drills: list[Drill]`
  (drills now hang off the issue they fix).
- `class Coaching(BaseModel)` — `summary: str`, `issues: list[Issue]`. (Drills live inside
  each `Issue`; a top-level `drills` property may be exposed if a consumer needs the flat
  list, but issues are the structure.)
- `def assemble_coaching(stroke_type: StrokeType, score: StrokeScore) -> Coaching`:
  - For each `factor` in `score.factors` with `sub_score < ISSUE_SUBSCORE_CUTOFF`, look up
    its `FaultEntry` and build an `Issue` (title/why/how/drills from the library).
  - **Priority** = rank by ascending `sub_score` (worst fault = priority 1).
  - A factor below cutoff with no library entry is skipped (and is a test-caught gap, not a
    crash) — every metric the pipeline emits MUST have a library entry.
  - `summary` is templated: states the overall score, names the top issue (or, when there
    are no faults, gives positive reinforcement naming the strongest metric).
  - Fully deterministic, offline, no cost. This object is the **source of truth.**

### 3. Optional voice pass — `coaching.py`

- `def build_voice_prompt(baseline: Coaching, score: StrokeScore) -> str` — instructs the
  model to rewrite **only** the `summary` and each issue's `why`/`how` in a warmer, more
  varied coaching voice, grounded in the supplied numbers; it must keep the same issues and
  the same drills verbatim and must not introduce new measurements. (Replaces the old
  free-generation `build_prompt`.)
- `def apply_voice(baseline, score, *, client, model) -> Coaching` — calls the
  OpenAI-compatible `client.chat.completions.parse(..., response_format=Coaching)` (default
  model `gemini-2.5-flash-lite`, env-configurable — unchanged from current setup) and
  returns the parsed rewrite.
- `def is_faithful(baseline: Coaching, rewrite: Coaching) -> bool` — the guard. Issues are
  matched by **`title`** (the voice prompt rewrites only `summary` and each issue's
  `why`/`how`, leaving `title` and `drills` verbatim, so `title` is a stable key):
  - same set of issue titles and same **count**;
  - within each matched issue, same drill `name`s — none added, removed, or renamed;
  - no number token (integer/decimal) appears anywhere in the rewrite's `summary` or any
    issue's `why`/`how` that is not present in the corresponding baseline text
    (cheap invented-number check).
  Returns `False` on any violation.
- `def generate_coaching(stroke_type, metrics, score, *, client=None, model=DEFAULT_COACHING_MODEL, use_voice=True) -> Coaching`:
  1. `baseline = assemble_coaching(stroke_type, score)`.
  2. If `not use_voice` → return `baseline`.
  3. If `client is None` → construct the default OpenAI-compatible client (as today). If
     construction or the call raises, or the result fails `is_faithful` → return `baseline`.
  4. Otherwise return the faithful rewrite.
  - Signature stays drop-in compatible with the pipeline (it passes
    `metrics`/`score`/`client`/`model`); `metrics` is accepted for compatibility, but
    assembly reads `score.factors` (the scored metrics).

### 4. Preview CLI — `cv-pipeline/src/tennis_cv/preview.py`

- `REPRESENTATIVE_PROFILES` — a fixed set of `StrokeScore`s exercising: shallow knee only,
  low contact only, bent arm only, all-good (no faults), and a multi-fault serve.
- `def render_profile(name, score, *, use_voice, client=None) -> str` — returns the full
  coaching text for one profile.
- `def main(argv=None) -> int` — prints each profile's coaching; deterministic by default,
  `--voice` to include the LLM pass (needs a key). Lets a human read real output and sign
  off the library.

## Error Handling & Edge Cases

- **Missing library entry** for a below-cutoff metric → that issue is skipped; a unit test
  asserts every pipeline metric has an entry, so this can't ship silently.
- **No faults** (all metrics ≥ cutoff) → `Coaching` with an empty `issues` list and a
  positive summary; the pipeline still returns a valid result.
- **LLM disabled / no key / network error / API error** → deterministic baseline returned.
- **Unfaithful rewrite** (drill changed, issue dropped, number invented) → baseline returned.
- The deterministic path never raises on valid `StrokeScore` input.

## Testing Strategy

- **`drills.py`:** every metric the metrics module emits (`knee_flexion`, `contact_height`,
  `arm_extension`) has a `FaultEntry` with ≥1 drill and non-empty `why`/`how`.
- **`assemble_coaching`:** below-cutoff metrics become issues; at/above become strengths;
  priority orders by ascending sub-score; all-good → no issues + positive summary; the
  summary names the top issue.
- **`is_faithful`:** accepts an identical/voice-only rewrite; rejects a dropped drill, a
  renamed drill, an added issue, and an invented number.
- **`generate_coaching`:** `use_voice=False` returns baseline; injected faithful client →
  rewrite; injected unfaithful client → baseline; injected raising client → baseline. All
  offline (client injected), as today.
- **`preview`:** `main([])` renders all profiles without error; output contains expected
  drill names.
- Existing pipeline tests keep passing (signature unchanged).

## Migration Notes

- `coaching.py`'s current free-generation `generate_coaching`/`build_prompt` are replaced.
  `Issue`/`Coaching` gain structure (`Issue.drills`); the pipeline's use of
  `result.coaching.summary` and the CLI's `coaching.model_dump()` continue to work.
- `Drill` moves to `drills.py` and is re-exported from `coaching` so existing imports
  (`from tennis_cv.coaching import Drill`) still resolve.

## Open Items (carried, not blocking)

1. **Library content is a v1 draft** — needs a real coach's review (the preview CLI exists
   for exactly this). Tracked as the "vetted drill library" follow-up.
2. **Metric target ranges remain placeholders** — calibration is separate work; until done,
   a fault may be mis-stated regardless of how good the drill text is.
