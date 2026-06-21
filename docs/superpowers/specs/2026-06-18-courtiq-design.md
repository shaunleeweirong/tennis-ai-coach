# Tennis AI Coach — Phase 0 (MVP) Technical Design

**Date:** 2026-06-18 (revised 2026-06-21)
**Status:** Approved direction — pending implementation plan
**Canonical product reference:** [`prd-tennis-ai-coach.md`](../../../prd-tennis-ai-coach.md)

> **Source of truth.** The PRD (`prd-tennis-ai-coach.md`) is the canonical
> product, architecture, and business document. This file is a focused
> **technical design for Phase 0 (the MVP)** derived from it. Where this file
> and the PRD ever disagree, **the PRD wins.** An earlier brainstorm version of
> this spec proposed an on-device / Supabase / serve-only design; that has been
> **superseded** by the PRD's server-side CV architecture.

## What changed from the brainstorm

The original brainstorm spec is replaced. The PRD overrides three core decisions:

| Topic | Brainstorm (superseded) | PRD (canonical) |
| --- | --- | --- |
| Where CV runs | On-device (Apple Vision) | **Server-side Python** (MediaPipe + ViTPose on GPU workers) |
| Backend stack | Supabase + one edge function | **FastAPI + Redis queue + GPU workers + S3 + Postgres** |
| MVP stroke scope | Serve only | **Single-stroke upload, classify forehand/backhand/serve/volley** |

The iOS app is now a **thin client**: it uploads video, polls for results, and
renders feedback. All computer vision and scoring run server-side, which keeps
the CV engine reusable when Android/web ship in Phase 2.

## Phase 0 scope (the MVP build)

Per the PRD's Phase 0 (months 0–3) and the user stories it covers:

- **US-001** Upload a clip (photo library or in-app record) → validated, with a
  friendly refilm error if unusable.
- **US-002** Stroke-by-stroke feedback: each detected stroke labelled, skeleton
  overlay at contact, ≥1 specific coaching note (metric available on tap).
- **US-003** Technique score (0–100) per stroke with 3–5 contributing factors;
  pro-form side-by-side comparison.
- **US-005** Progress history + trend chart per stroke type.
- **US-008** Subscription paywall (StoreKit 2), free-tier limit enforced.

**Deferred to later phases (not built in Phase 0):** conversational AI coach
(US-004, Phase 1), match/practice statistics + ball/court tracking (US-006,
Phase 1+), coach/club tier (US-007, Phase 2), shareable highlight export
(Phase 1), Android/web (Phase 2).

## Architecture (Phase 0)

Four layers, exactly as the PRD describes:

1. **iOS client (Swift 6 / SwiftUI, min iOS 17)** — record/select video,
   request presigned upload URL, upload to object storage, start an analysis
   job, poll job status, render results. SwiftData caches analyses/history for
   offline viewing. StoreKit 2 for subscriptions. URLSession async/await.
   *Light mode only* (per PRD Design Considerations).
2. **API / orchestration (Python FastAPI)** — auth/session, presigned uploads,
   job creation + status, results fetch, history, subscription verification.
3. **GPU worker CV pipeline (Python, PyTorch)** — pose estimation
   (**MediaPipe BlazePose** for the wedge, **ViTPose** for high-accuracy joints;
   both confirmed Apache-2.0 / commercial-safe in the PRD's validation log),
   stroke detection + classification, technique-metric computation, 0–100
   scoring, skeleton-overlay rendering, then an LLM call (Claude) for grounded
   coaching feedback.
4. **Data / external** — Postgres (users, analyses, stroke scores, overlays,
   subscriptions), S3-compatible object storage (raw video + rendered
   overlays), Redis queue (async jobs), Claude API (coaching text).

### Phase 0 data flow (single-stroke path)

1. User selects/records a clip in iOS and taps Analyse.
2. App gets a presigned upload URL, uploads the video to object storage.
3. App POSTs an analysis job (`mode: "stroke"`); backend enqueues it, returns a
   `job_id`.
4. A GPU worker runs pose estimation → stroke detect/classify → technique
   metrics → 0–100 score.
5. Worker renders skeleton-overlay key frames to storage.
6. Worker calls Claude with the computed metrics → grounded coaching feedback +
   drill.
7. Worker writes results + scores + overlay URLs to Postgres; marks job done.
8. App polls `GET /jobs/{job_id}`, fetches the analysis, renders it; SwiftData
   caches it to history.

**Guiding principle (unchanged from the PRD):** the CV pipeline computes the
measurable, deterministic technique metrics and the 0–100 score; the LLM is the
grounded "coach's voice" on top — it explains and recommends, never measures.

## API surface (Phase 0 subset)

From the PRD's internal API, the endpoints Phase 0 needs:

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/uploads` | Presigned video upload URL |
| POST | `/api/v1/analyses` | Start a stroke analysis job |
| GET | `/api/v1/jobs/{jobId}` | Poll job status |
| GET | `/api/v1/analyses/{id}` | Fetch a completed analysis |
| GET | `/api/v1/analyses` | List analysis history |
| DELETE | `/api/v1/analyses/{id}` | Delete an analysis + its video |
| POST | `/api/v1/subscriptions/verify` | Verify StoreKit receipt, set entitlement |

(Coach and chat endpoints from the PRD are out of Phase 0 scope.)

## Data model (Phase 0)

Postgres is the source of truth for app data; Apple owns subscription billing
truth (entitlement is cached). Object storage owns video/overlay blobs.

- `users`
- `analyses` — per-clip results, `stroke_type`, timestamps, status
- `stroke_scores` — 0–100 score + contributing-factor breakdown per stroke
- `overlays` — references to rendered skeleton-overlay frames in storage
- `subscriptions` — tier + entitlement, restored across devices via backend

## Error handling & edge cases (Phase 0)

- Unusable video (too dark, wrong angle, no player detected) → validated at the
  backend **before** GPU/LLM cost; specific, friendly refilm guidance (US-001).
- Low pose confidence → analysis proceeds, feedback flagged lower-confidence.
- LLM/network failure on the worker → retain score + metrics + overlays; mark
  coaching text as retryable.
- Job failure → surfaced through `GET /jobs/{jobId}` with a reason.

## Privacy & data retention (Phase 0)

Video contains identifiable people, so: explicit consent to process video, a
clear privacy policy, user-initiated deletion (FR-12, `DELETE /analyses/{id}`),
and a retention policy that **auto-deletes raw clips after N days while keeping
metrics + overlays** (Open Question #7 — N to be set during planning).

## Testing strategy (Phase 0)

- **Worker unit tests:** metric math + scoring against known pose fixtures
  (deterministic, no network).
- **Pipeline tests:** sample clips → expected stroke labels, phase boundaries,
  and score ranges.
- **API tests:** job lifecycle (create → poll → fetch), auth scoping, schema
  validation of Claude responses (mocked).
- **iOS tests:** upload/poll/render flow against a mocked backend; SwiftData
  cache + offline viewing.

## Open items carried from the PRD (resolve during planning)

1. **Ball/court licensing** — not a Phase 0 blocker (Phase 0 is pose-only;
   MediaPipe + ViTPose are cleared). Resolve before Phase 1.
2. **GPU host** (Modal / Replicate / RunPod / raw cloud) — drives cost-per-
   analysis and the free-tier limit.
3. **On-device MediaPipe preview** — defer to Phase 1 vs include now.
4. **Free-tier shape** — hard paywall+trial vs freemium (A/B later).
5. **Filming standardisation** — how strictly to enforce camera angle.
6. **Pro-form benchmark source** — where US-003's reference comparison footage
   comes from (rights-sensitive).
7. **Raw-video retention window** — set N.
8. **Missing asset** — the PRD references `tennis-ai-coach-architecture.svg`,
   which is not yet in the repo.
