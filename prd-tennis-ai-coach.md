# PRD: Tennis AI Coach — Upload-a-Clip Stroke Analysis App

## Introduction

An iOS app that lets recreational and competitive tennis players upload a video of their gameplay (or a single stroke) and receive AI-powered feedback on how to improve, plus performance statistics. The core wedge is **biomechanical stroke coaching from an uploaded clip** — analysing the player's body movement (kinetic chain, contact point, swing path) and returning specific, actionable feedback. This deliberately sidesteps the entrenched leader (SwingVision), which is iOS-strong but match-stats-first and shallow on stroke biomechanics.

All heavy computer vision runs **server-side** in Python, so the iOS app is a thin client that uploads video, polls for results, and renders feedback. This keeps the app simple and means the CV engine is reusable when Android and web ship later.

**Primary business goal:** reach **$5,000/month recurring revenue** (~400–650 paying subscribers depending on price), led by organic acquisition and a coach/club B2B tier rather than paid consumer ads.

---

## Goals

- Let a user upload a tennis clip and get stroke-by-stroke biomechanical feedback within a few minutes.
- Detect and classify strokes (forehand, backhand, serve, volley) and score technique against pro-form benchmarks.
- Provide a clear "here's your weakness, here's the drill to fix it" output, not just raw numbers.
- Track progress over time so users see improvement (the retention hook).
- Ship a coach/club tier so one paying coach brings many players (low-CAC growth).
- Hit $5,000/month within 9–18 months on a lean, organic-led model.

---

## User Stories

### US-001: Upload a clip for analysis
**Description:** As a player, I want to upload a video of my gameplay so that I can get feedback on my technique.

**Acceptance Criteria:**
- [ ] User can select a video from their photo library or record in-app
- [ ] Supported: single-stroke clips and full-rally/practice clips up to a defined length (e.g. 3 min for free, longer for paid)
- [ ] Upload shows progress and a clear "analysing" state with estimated time
- [ ] If the video is unusable (too dark, wrong angle, no player detected), the user gets a specific, friendly error explaining how to refilm

### US-002: Receive stroke-by-stroke feedback
**Description:** As a player, I want feedback on each stroke so that I know exactly what to fix.

**Acceptance Criteria:**
- [ ] Each detected stroke is labelled (forehand / backhand / serve / volley)
- [ ] Each stroke shows a skeleton overlay on the video frame at contact
- [ ] Each stroke returns at least one specific technique note (e.g. "contact point is behind your front foot — try to meet the ball further forward")
- [ ] Feedback is phrased as coaching, not raw joint angles, with the underlying metric available on tap

### US-003: See a technique score and benchmark
**Description:** As a player, I want a score compared to good form so that I can track where I stand.

**Acceptance Criteria:**
- [ ] Each stroke type gets a 0–100 technique score
- [ ] Score is explained by 3–5 contributing factors (kinetic chain timing, contact point, follow-through, balance, racquet path)
- [ ] A side-by-side comparison to a pro/reference form is available

### US-004: Conversational AI coach
**Description:** As a player, I want to ask follow-up questions about my feedback so that I understand how to improve.

**Acceptance Criteria:**
- [ ] After analysis, user can ask free-text questions (e.g. "why is my serve scored low?")
- [ ] The coach answers using the actual analysis data from that clip, not generic advice
- [ ] Coach suggests 1–2 specific drills tied to the detected weakness

### US-005: Track progress over time
**Description:** As a player, I want to see my scores over time so that I stay motivated.

**Acceptance Criteria:**
- [ ] Each analysis is saved to the user's history
- [ ] A trend chart shows technique score per stroke type over time
- [ ] User can compare any two analyses side by side

### US-006: Match/practice statistics (Phase 1+)
**Description:** As a player, I want stats from a full clip so that I understand my game beyond single strokes.

**Acceptance Criteria:**
- [ ] From a full-court clip: shot count by type, rally length distribution, placement heat map
- [ ] Stats derived from ball + court tracking (server-side)
- [ ] Clearly flagged as "beta" while accuracy on phone footage is being tuned

### US-007: Coach/club tier (Phase 2)
**Description:** As a coach, I want to review my players' uploads so that I can coach remotely and scale.

**Acceptance Criteria:**
- [ ] A coach account can invite players via a link/code
- [ ] Coach sees a dashboard of all linked players' analyses
- [ ] Coach can leave annotations/notes on a player's analysis
- [ ] Billing: coach pays a flat tier; linked players use a free or discounted plan

### US-008: Subscription paywall
**Description:** As the business, I need to gate premium analysis so that the app generates revenue.

**Acceptance Criteria:**
- [ ] Free tier has a clear limit (e.g. N analyses/month or single-stroke only)
- [ ] Paywall offers monthly and annual plans, annual visibly discounted
- [ ] Purchases processed via Apple In-App Purchase (StoreKit 2)
- [ ] Subscription state restored across devices via the backend

---

## Functional Requirements

- FR-1: The iOS app must let users record or select a video and upload it to the backend.
- FR-2: The backend must validate the video (resolution, length, player detectable) and reject with a specific reason if unusable.
- FR-3: The backend must run pose estimation on the video and detect/classify strokes.
- FR-4: For each stroke, the system must compute technique metrics (contact point, kinetic-chain timing, swing path, balance) and produce a 0–100 score.
- FR-5: The system must generate natural-language coaching feedback per stroke via an LLM, grounded in the computed metrics.
- FR-6: The system must return a skeleton overlay rendered on key frames.
- FR-7: The app must let users ask follow-up questions answered from the clip's analysis data.
- FR-8: The system must persist every analysis to the user's history and render a progress trend chart.
- FR-9 (Phase 1+): The backend must run ball tracking + court detection on full-court clips to produce match statistics.
- FR-10 (Phase 2): The system must support coach accounts that link to multiple player accounts with a shared dashboard.
- FR-11: The app must enforce free-tier limits and process subscriptions via StoreKit 2.
- FR-12: The backend must store video and analysis securely and allow the user to delete their data.

---

## Non-Goals (Out of Scope)

- **No automated match scoring or electronic line calling.** This is SwingVision's data-moat strength (trained on 500M+ shots); competing here is not viable for a newcomer.
- **No live/real-time on-court analysis in v1.** The product is upload-then-analyse.
- **No Android or web app in v1.** Documented as Phase 2; the server-side CV engine is built platform-agnostic so this is a client-only effort later.
- **No wearable or hardware sensor integration.** The market has moved to camera-based CV.
- **No tournament-grade ball-tracking accuracy claims.** Phone-footage stats ship as "beta."
- **No social feed / community in v1** (a sharing/export feature is in, a full social network is not).

---

## Design Considerations

- **Light mode for all screens and charts.** No dark theme.
- Onboarding must get the user to their first "aha" analysis fast — most trial cancellations happen on day 0, so the first upload must succeed and impress.
- Filming guidance is critical: an in-app guide showing the correct camera angle (side-on for strokes, elevated back-of-court for match stats) materially improves CV accuracy and reduces "unusable video" errors.
- Feedback must read like a coach talking, with the numbers tucked behind a tap. Leading with joint angles will feel clinical and lose casual users.
- Progress charts and shareable highlight clips are the retention and virality hooks — design for export to TikTok/Instagram/YouTube vertical format.

---

## Tech Architecture (High-Level)

This is a **hybrid mobile + backend** system. The iOS app is thin; the CV/ML lives server-side.

### Why iOS-first (decision rationale)
- iOS users spend roughly 2–3x more on subscriptions — decisive when the goal is $5K MRR from a few hundred subscribers.
- One hardware target (a handful of iPhone models) vs thousands of Android devices cuts camera/performance testing dramatically.
- The builder already ships iOS/Swift (Strength Tracker), so it's the lowest-friction path.
- **The CV engine is server-side Python, so platform choice does not affect the tracking stack at all.** None of the tracking tools are iOS-only. Android/web later reuse the same backend.

### Mobile stack (iOS)
- Language: Swift 6 / SwiftUI
- Min iOS: 17.0
- Persistence: SwiftData (local cache of analyses, history)
- Networking: URLSession async/await; multipart upload for video
- Payments: StoreKit 2 (auto-renewable subscriptions)
- On-device (optional, Phase 1): MediaPipe BlazePose via its iOS SDK for instant rough skeleton preview while the full server analysis runs
- Auth: token-based (e.g. Clerk or Sign in with Apple → backend session)

### Backend stack
- API + orchestration: Python (FastAPI) — same language as the ML, simplest integration
- Job queue: Redis + a worker queue (Celery/RQ, or BullMQ if a Node gateway is preferred) — video analysis is async and can take seconds to minutes
- ML runtime: PyTorch, run on GPU instances (the CV models need GPU for reasonable speed)
- Object storage: S3-compatible bucket for uploaded video and rendered overlays
- Database: PostgreSQL (users, analyses, scores, coach-player links, subscription state)
- LLM: hosted API (e.g. Claude) for grounded coaching feedback and the conversational coach
- Hosting: a GPU-capable host (e.g. Modal, Replicate, RunPod, or a cloud GPU instance) for the ML workers; standard host for the API

### Computer vision components (all server-side)

License statuses below were **validated against primary sources on 2026-06-20** (repo LICENSE files / Hugging Face model cards), not assumed from memory. See the Tooling Validation Log at the end of this PRD for the source of each status.

| Component | Tool | License status (validated 2026-06-20) | Role |
|---|---|---|---|
| Pose estimation (MVP wedge) | **MediaPipe BlazePose** | ✅ **Apache 2.0 — commercial OK.** Confirmed at `github.com/google-ai-edge/mediapipe/LICENSE` | 33 body landmarks; stroke biomechanics; on-device preview possible |
| Pose estimation (high accuracy) | **ViTPose** (usyd-community checkpoints) | ✅ **Apache 2.0 — commercial OK.** Checkpoints tagged `license: apache-2.0` on HF; training data permissive (COCO CC-BY-4.0, MPII BSD) | Server-side high-accuracy joints for the detailed paid analysis |
| Stroke classification | Custom (pose landmarks → LSTM/classifier) | ✅ Your own first-party code | Forehand/backhand/serve/volley labelling |
| Court detection | **yastrebksv/TennisCourtDetector** | ⚠️ **No LICENSE file = all rights reserved.** Reference architecture only — retrain weights or get permission before commercial use | 14 court keypoints → homography → bird's-eye view |
| Ball tracking (Phase 1+) | **yastrebksv/TrackNet** (TrackNet architecture) | ⚠️ **No LICENSE file = all rights reserved.** Reference only (confirmed: repo has no license, 205★, 10 commits) | Ball position per frame for stats |
| Bounce detection (Phase 1+) | **yastrebksv/TennisProject** (CatBoost on trajectory) | ⚠️ **No LICENSE file = all rights reserved.** Reference only | Bounce points for placement/depth |
| Full reference pipeline | **ArtLabss/tennis-tracking** | ⚠️ Verify license before use; last meaningful update early 2024 | Reference architecture only |
| Ball tracking (buy option) | **SportAI API** | 💲 **PAID**, quote-based B2B commercial licence | Optional later: outsource ball tracking if own models underperform on phone footage |

**Cost summary:** Every model needed for the **MVP (pose-only) is free AND commercially licensed (Apache 2.0)** — MediaPipe and ViTPose are both cleared. The Phase 1 ball/court tracking repos are free to download but **not** commercially licensed (no LICENSE file), so they are reference-only until you retrain weights or secure permission. The only paid CV option is **SportAI's hosted API** (optional, Phase 1+ buy-vs-build). Your real recurring costs are infrastructure (GPU compute, storage) and the LLM API — not the tracking models.

**Licensing decision (validated, not assumed):** "Free to download" ≠ "licensed for commercial use." Confirmed findings:
- **MediaPipe — Apache 2.0.** Unambiguously safe for a paid app. Verified at the Google repo.
- **ViTPose (usyd-community) — Apache 2.0.** The actual checkpoints you'd load are Apache-2.0-tagged and trained on permissively licensed datasets (COCO CC-BY-4.0, MPII BSD). Safe to ship. (Note: the *Hugging Face Transformers* integration code is also Apache 2.0; what matters for shipping is the checkpoint licence, which is confirmed clear.)
- **yastrebksv TrackNet / TennisCourtDetector / TennisProject — NO licence.** I fetched the TennisProject repo directly and confirmed there is no LICENSE file. Under copyright law, no licence means **all rights reserved by default**. Before any commercial use you must (a) get written permission from the author, (b) wait for a permissive licence to be added, or (c) **retrain your own weights** on openly licensed data using the architecture as a reference (architectures are not copyrightable; only the specific code and trained weights are).

**Net effect on the plan:** the **MVP wedge (pose-based stroke coaching) ships with zero licensing risk.** Ball/court tracking — the only components with a legal asterisk — is deferred to Phase 1, by which point the retrain-vs-license-SportAI decision is made. This is a tracked pre-launch blocker (see Open Questions #1), not a footnote.

### Data flow (single-stroke analysis — the MVP path)
1. User records/selects a clip in the iOS app and taps Analyse.
2. App requests a presigned upload URL from the backend, uploads the video to object storage.
3. App POSTs an analysis job referencing the uploaded video; backend enqueues it and returns a `job_id`.
4. A GPU worker picks up the job: runs pose estimation (MediaPipe/ViTPose), detects and classifies strokes, computes technique metrics, scores each stroke.
5. Worker renders skeleton-overlay key frames to storage.
6. Worker calls the LLM with the computed metrics to generate grounded coaching feedback and drills.
7. Worker writes results + scores + overlay URLs to Postgres and marks the job done.
8. App polls `GET /jobs/{job_id}` (or receives a push), then fetches and renders the analysis; SwiftData caches it to history.

### Offline behaviour
- Past analyses are cached in SwiftData and viewable offline.
- New uploads require a network connection (analysis is server-side).

### Architecture diagram
A full validated architecture diagram is provided alongside this PRD as `tennis-ai-coach-architecture.svg` (light mode). It shows the four layers (Client → API/Orchestration → GPU Worker CV Pipeline → Data/External), the numbered request flow, and the validated licensing status colour-coded per component (green = commercial-safe, amber = licence unresolved, paid flagged separately).

Text summary of the same flow:
```
[iOS App] --upload--> [Object Storage]
    |                       ^
    | POST /analyses        |
    v                       |
[FastAPI] --enqueue--> [Redis Queue] --> [GPU Worker]
    |                                       | pose (MediaPipe + ViTPose, Apache 2.0)
    | poll /jobs/{id}                       | ball/court (Phase 1, reference-only)
    v                                       | + LLM feedback
[Postgres] <------- results -------- [GPU Worker] --overlays--> [Object Storage]
```

---

## API Endpoints

### External APIs

#### SportAI API (OPTIONAL — Phase 2+ buy-vs-build only)
- **Docs:** https://www.sportai.com (checked on 2026-06-20; confirm current API docs/pricing directly with SportAI — they are B2B and pricing is quote-based)
- **Base URL:** TBD (provided on commercial agreement)
- **Auth:** API key (expected)
- **Rate limit:** Per commercial agreement
- **Pricing:** **Paid**, quote-based B2B licensing
- **Role:** Camera-agnostic technique/ball analysis; claims ~98% precision vs Hawk-Eye from a standard 30fps/1080p phone clip. Only adopt if self-built ball tracking underperforms on real user footage.

#### LLM API (e.g. Claude) — coaching feedback + conversational coach
- **Docs:** https://docs.claude.com (checked on 2026-06-20)
- **Auth:** API key (bearer)
- **Pricing:** Paid, per-token
- **Role:** Generate grounded coaching feedback from computed metrics; power US-004 conversational coach.

*The open-source CV models (MediaPipe, TrackNet, TennisCourtDetector, ViTPose) are self-hosted, not external APIs — they run inside your own workers.*

### Internal API (first-party, defined by this PRD)

| Method | Path | Purpose | Auth | Request body | Response |
|--------|------|---------|------|--------------|----------|
| POST | /api/v1/uploads | Get presigned video upload URL | Session | `{filename, contentType}` | `{uploadUrl, videoId}` |
| POST | /api/v1/analyses | Start an analysis job | Session | `{videoId, mode: "stroke"\|"match"}` | `{jobId, status}` |
| GET | /api/v1/jobs/{jobId} | Poll job status | Session | — | `{status, progress, analysisId?}` |
| GET | /api/v1/analyses/{id} | Fetch a completed analysis | Session | — | `{strokes[], scores, overlays[], feedback}` |
| GET | /api/v1/analyses | List user's analysis history | Session | query: `page,limit` | `Analysis[]` |
| POST | /api/v1/analyses/{id}/chat | Ask the AI coach a follow-up | Session | `{message}` | `{reply, drills[]}` |
| DELETE | /api/v1/analyses/{id} | Delete an analysis + its video | Session | — | `{deleted: true}` |
| POST | /api/v1/coach/invites | Coach creates a player invite | Coach session | `{}` | `{inviteCode, link}` |
| POST | /api/v1/coach/link | Player joins a coach via code | Session | `{inviteCode}` | `{coachId}` |
| GET | /api/v1/coach/players | Coach lists linked players | Coach session | — | `Player[]` |
| POST | /api/v1/coach/notes | Coach annotates a player analysis | Coach session | `{analysisId, note}` | `{noteId}` |
| POST | /api/v1/subscriptions/verify | Verify StoreKit receipt, set entitlement | Session | `{receipt}` | `{tier, expiresAt}` |

---

## Data Dependencies

- **Reads from:** uploaded user video (object storage); pretrained CV model weights (self-hosted); LLM API (feedback generation); StoreKit receipts (subscription state).
- **Writes to:** `users`, `analyses` (per-clip results), `stroke_scores`, `overlays` (frame image refs), `coach_player_links`, `coach_notes`, `subscriptions` tables; rendered overlay images to object storage.
- **Freshness:** Analysis is near-real-time from the user's view (seconds to a few minutes per clip). Subscription state must be current. Progress history is append-only.
- **Volume estimate:** Roughly 5–30 analyses per active user per month; each raw clip tens to a few hundred MB. Storage and GPU compute are the main cost drivers — plan a retention/cleanup policy (e.g. delete raw video after N days, keep overlays + metrics).
- **Sensitive data:** Video of identifiable people (the user, possibly hitting partners and bystanders) — privacy-relevant. Email/account (PII). Payment handled by Apple (you don't store card data). Need a clear privacy policy, consent for processing video, and user-initiated deletion (FR-12).
- **Source of truth:** Your Postgres owns analyses, scores, and coach links. Apple owns subscription billing truth (you cache entitlement). Object storage owns video/overlay blobs.

---

## Phased Roadmap (tying back to the $5K/month goal)

**Phase 0 — MVP (months 0–3):** iOS-only. Single-stroke upload → pose estimation (MediaPipe + ViTPose) → technique score + grounded coaching feedback + drill. Progress history. Paywall (annual-pushed, ~$9.99/mo or ~$79–99/yr). No ball/court tracking yet. Goal: prove the feedback feels genuinely useful (target ~90%+ "this was helpful" in testing) before building anything harder.

**Phase 1 — Differentiate + monetise (months 3–9):** Conversational AI coach; serve analysis (high-demand, static like a golf swing); shareable vertical highlight export. Begin ball + court tracking in beta for basic match stats. Focus stays on the individual consumer subscriber — no coach/club tier yet.

**Phase 2 — Scale (months 9–18):** **Coach/club B2B tier** (the highest-leverage growth lever — 10–15 paying coaches can underpin most of $5K MRR at low CAC); full match stats (rally length, placement heat maps) from full-court clips; consider licensing SportAI if self-built ball tracking can't handle phone footage; Android + web clients (reusing the same backend); leaderboards/light social.

**Revenue math reminder:** ~400–650 net paying subscribers clears $5K/month depending on price. The constraint is acquisition, not the headcount — lean ~70% on organic/ASO/creator content (and, once it launches in Phase 2, the coach tier); avoid Sports-category paid ads (most expensive ASA vertical, ~$14–20 CPA) until a channel demonstrably clears LTV:CAC > 3:1.

---

## Success Metrics

- **Product:** First-upload success rate > 85% (video accepted and analysed). Median analysis turnaround < 3 minutes.
- **Feedback quality:** ≥ 90% of test users rate stroke feedback "helpful" before scaling spend.
- **Activation:** ≥ 40% of new users complete a second analysis within 7 days.
- **Monetisation:** Trial-to-paid > 25%; monthly churn < 5%; reach $5,000 MRR within 9–18 months.
- **Coach channel (Phase 2):** ≥ 3 paying coaches within the first 90 days of the coach tier launching (PMF signal for B2B).
- **Stability:** Crash-free sessions > 99%.

---

## Open Questions

1. **Licensing (validated — partial blocker remains):** Validation complete as of 2026-06-20. **MediaPipe and ViTPose are both confirmed Apache 2.0 and cleared for commercial use**, so the pose-only MVP has no licensing blocker. The remaining blocker is Phase 1 only: the yastrebksv ball/court repos have no licence (all rights reserved). Decision needed *before Phase 1 ships*, not before MVP: retrain own weights on openly licensed data, contact the author for permission, or license SportAI's API instead. **Recommended: ship the MediaPipe + ViTPose MVP now; resolve ball/court licensing during Phase 1.**
2. **GPU hosting cost model:** Which host (Modal / Replicate / RunPod / raw cloud GPU) gives the best cost-per-analysis at low volume? This drives unit economics and the free-tier limit.
3. **On-device preview:** Is the MediaPipe on-device skeleton preview worth the extra iOS work in v1, or defer to Phase 1?
4. **Free-tier shape:** Hard paywall + short trial vs generous freemium — A/B test which converts better in this category (hard paywall historically converts higher in health & fitness).
5. **Filming standardisation:** How strictly to enforce camera angle? A required in-app filming guide vs accepting messy footage and tuning models to tolerate it — affects accuracy and drop-off.
6. **Pro-form benchmark source:** Where do the "good form" reference comparisons come from (licensed pro footage vs your own coached reference clips)? Licensing pro footage has its own rights issues.
7. **Data retention:** How long to keep raw user video given storage cost and privacy — propose auto-deleting raw clips after N days while keeping metrics + overlays.

---

## Go-To-Market Plans to $5,000 MRR

### The target, in plain numbers

$5,000/month net of Apple's cut (~15% under the Small Business Program) means you need roughly **$5,880 in gross subscription revenue**. Translated to subscribers:

| Price point | Net paying subs needed | Gross subs (pre-fee) |
|---|---|---|
| $7.99/mo | ~735 | ~625 |
| $9.99/mo | ~590 | ~500 |
| $12.99/mo | ~450 | ~385 |
| $14.99/mo (or ~$149/yr) | ~395 | ~335 |
| Coach tier $39/mo | ~150 coaches alone | — |

The headcount is small. The whole game is **acquisition efficiency** — getting to a few hundred payers without burning cash in the most expensive paid-ads category in the App Store (Sports, ~$14–20 CPA per install before you even get to trial conversion). Every plan below is built to avoid that trap.

A useful framing: at a ~25% trial-to-paid rate and ~5% monthly churn, ~500 active payers requires roughly **2,000 trials started** and a steady top-of-funnel because you're replacing churned users every month. Once you internalise that, the plans are just different engines for generating those trials cheaply.

---

### Plan A — Coach-Led (B2B2C) — *recommended primary*

**Thesis:** One coach brings 10–40 students. Coaches have low churn (it's a work tool), high willingness to pay, and they market the app *for* you to a captive, motivated audience. This is the single highest-leverage, lowest-CAC path for a solo builder, and it's exactly where SwingVision is weakest (they're consumer-match-stats-first).

**How it works:**
- Coach pays a flat tier (~$39–79/mo) for a dashboard to review students' uploads, leave annotations, and track progress.
- Linked students get a free or discounted plan — the coach is the payer and the distributor.
- You sell to the coach, not the player. Coaches are findable, reachable, and concentrated.

**The math:** ~80–130 paying coaches at $39–79/mo clears $5K MRR on its own. Even a blended model where coaches drive student upgrades gets you there with fewer direct sales.

**Channels:**
- Direct outreach to independent coaches and small academies (LinkedIn, local club directories, USPTA/PTR coach networks, your own city's clubs first).
- A "coach ambassador" program: free lifetime tier for the first 20 coaches who onboard ≥10 students and give feedback.
- Partnerships with junior development programs and high-school/college teams (one coach = a whole roster).
- Show up where coaches already are: r/tennis coaching threads, coaching Facebook groups, PTR/USPTA forums and events.

**KPIs / kill criteria:** Land **3 paying coaches in 90 days** or the B2B thesis is weak and you pivot weight to Plan B. Target 15 paying coaches by month 6.

**Pros:** Lowest CAC, lowest churn, fastest path to the number, defensible (coach relationships + stored student history are switching costs).
**Cons:** Slower to start (sales cycle, relationship-building), requires the coach dashboard (US-007, a Phase 2 build) rather than being available at MVP — so it's a later-stage lever, not a launch one.

---

### Plan B — Organic Content Engine (Creator-Led) — *recommended co-primary*

**Thesis:** Tennis improvement content is a massive, hungry niche on YouTube/TikTok/Instagram. The category's growth is creator-driven. You don't need to buy attention — you can earn it, and the app *is* the content (skeleton overlays and "here's what's wrong with your forehand" are inherently watchable).

**How it works:**
- Publish short-form teardown content: take a viral/amateur tennis clip, run it through the app, narrate the fix. The output is the hook.
- Each video CTAs to "analyse your own stroke." App exports shareable vertical clips with your watermark → built-in viral loop.
- Long-form on YouTube for depth and SEO ("why your serve has no power — analysed frame by frame").

**The math:** Organic drives ~70% of non-gaming subscription revenue industry-wide. A single video that lands can drive thousands of installs at $0 CAC. You need a handful of hits, not a hit every week.

**Channels:**
- Your own TikTok/IG/YouTube/Shorts under a tennis-improvement brand.
- Partner with mid-size tennis creators (25K–150K subs) — gift them the Pro tier, co-create a "I analysed my subscribers' strokes" video. Far cheaper than ads.
- Reddit (r/10s, ~193K weekly visitors): genuinely useful teardown posts, not spam.
- ASO: the app store itself is a search engine — optimise for "tennis swing analysis," "forehand fix," "tennis AI coach."

**KPIs / kill criteria:** One video > 50K views within 60 days, or rework the content format. Target a repeatable format producing ≥1 install-driving video/week by month 3.

**Pros:** $0–low CAC, compounding (old videos keep converting), builds a brand moat and an audience you own.
**Cons:** Slow and non-linear, requires consistent output and on-camera/editing effort, hit-driven (variance is high).

---

### Plan C — Niche Domination (Wedge-First)

**Thesis:** Don't launch "for all tennis players." Own one underserved, searchable niche completely, become the obvious default there, then expand. Beating SwingVision everywhere is impossible; beating it for *one specific group* is very doable.

**Candidate niches:**
- **Juniors + tennis parents** — parents pay readily for kids' development, film matches already, and cluster in junior-circuit communities.
- **The serve** — the most-requested, most-frustrating stroke, and biomechanically static (like a golf swing), so your pose-only MVP nails it. "The serve app."
- **One-handed backhand players** — small, passionate, identity-driven community.
- **A single country/language** SwingVision underserves (you have an APAC base — Singapore/SEA tennis communities are reachable and less saturated).

**How it works:** Build features, content, and benchmarks specifically for the niche. Be the #1 result and the community-recommended tool for that one thing.

**The math:** A niche of even a few hundred thousand searchers globally is more than enough for 500 payers. Concentration beats breadth at this scale.

**KPIs / kill criteria:** Become a recommended tool in ≥2 niche communities within 90 days. If the niche can't produce 50 payers, it's too small — pick another.

**Pros:** Cheap, focused, defensible, clear messaging, easy word-of-mouth within a tight community.
**Cons:** Caps your ceiling until you expand; picking the wrong niche costs time.

---

### Plan D — Paid Acquisition (Use Sparingly / Late)

**Thesis:** Paid ads are a *scaling* tool once you have proven conversion and LTV — not a cold-start engine. In the Sports category they will bleed you dry if used too early.

**Reality check:** Sports is the most expensive Apple Search Ads vertical (~$14–20 CPA per install in 2025 data). After trial-funnel drop-off, your cost *per paying subscriber* is a multiple of that. You cannot brute-force $5K MRR profitably this way as a bootstrapper.

**When it earns its place:**
- Only after a channel demonstrably clears **LTV:CAC > 3:1**.
- Use small, targeted ASA bursts to **seed keyword rankings** in 1–2 countries, then let organic compound (the paid→organic flywheel).
- Retarget warm users (people who installed but didn't convert) rather than buying cold installs.

**KPIs / kill criteria:** Any campaign that can't show 3:1 LTV:CAC within its measurement window gets killed. No exceptions early on.

**Pros:** Predictable and scalable *once economics are proven*; good for pouring fuel on a fire that's already lit.
**Cons:** Most expensive category in the store; murders bootstrapped runway if used as the primary engine.

---

### Recommended sequence (how the plans combine)

These aren't mutually exclusive — they stack. The realistic path to $5K MRR:

1. **Months 0–3 (MVP + wedge):** Launch MediaPipe + ViTPose MVP focused on **one niche (Plan C)** — the serve or juniors is the strongest opener. Start the **content engine (Plan B)** day one; the app's overlays are your content.
2. **Months 3–9 (consumer monetisation):** Differentiate with the conversational AI coach, serve analysis, and shareable highlight export. Drive consumer subs hard through the content engine (Plan B) and niche word-of-mouth (Plan C). No coach/club tier yet — this window is about proving consumer trial-to-paid and retention.
3. **Months 9–18 (B2B turn-on + scale):** Ship the coach dashboard and run **Plan A** hard — direct outreach + ambassador program. Once trial-to-paid and LTV are proven, layer in **Plan D** paid bursts to amplify the winning organic channels, and broaden out of the launch niche.

**Sequencing note:** the coach/club tier is now a **Phase 2 (months 9–18)** lever, not an early one. This is a deliberate trade-off — it pushes the single lowest-CAC channel later, so the path to $5K MRR leans harder on consumer subscriptions (organic content + niche) in months 3–9. If consumer acquisition proves slow in that window, pulling the coach tier forward is the most obvious lever to reconsider.

**Most likely revenue mix at $5K MRR:** because the coach tier is now a Phase 2 lever, the *initial* path to $5K is **consumer-led** — the bulk from consumer subscriptions driven by organic content (Plan B) and niche word-of-mouth (Plan C), with paid (Plan D) only amplifying once proven. Concretely, roughly **450–650 consumer subscribers** gets you across the line without the coach tier at all. The coach/club tier (Plan A), once it launches in Phase 2, then becomes the highest-leverage way to *grow past* $5K and de-risk consumer churn — a smaller mix of paying coaches can replace a large chunk of consumer headcount at lower CAC.

### GTM success metrics

- **CAC by channel** tracked separately — coach, organic, niche, paid — kill anything that can't clear LTV:CAC > 3:1.
- **Blended CAC < $30** (and ideally near-zero for coach + organic).
- **Coach channel (Phase 2, from month 9):** 3 paying coaches within 90 days of the coach tier launching; 15 within 6 months of launch.
- **Content:** a repeatable install-driving format by month 3; one >50K-view video by month 2.
- **Trial-to-paid > 25%**, monthly churn < 5%, annual-plan mix > 50% of new subscriptions.
- **$5K MRR within 9–18 months**, reached consumer-first; the coach tier (Phase 2) is the lever to sustain and grow past it.

---

## Appendix: Tooling Validation Log (checked 2026-06-20)

Every tool below was checked against a primary source on this date rather than assumed. Re-verify before commercial launch, as licences and repos change.

| Tool / model | What was confirmed | Source | Verdict |
|---|---|---|---|
| **MediaPipe BlazePose** | Released by Google under Apache 2.0; explicitly permits commercial use, modification, distribution | `github.com/google-ai-edge/mediapipe/blob/master/LICENSE`; corroborated by HF model cards (opencv/pose_estimation_mediapipe) | ✅ Ship it |
| **ViTPose (usyd-community checkpoints)** | Checkpoints tagged `license: apache-2.0`; training datasets permissive (COCO CC-BY-4.0, MPII BSD) | HF model cards: `usyd-community/vitpose-base`, `vitpose-base-simple`, `vitpose-plus-*` | ✅ Ship it |
| **HF Transformers (ViTPose integration code)** | Library is Apache 2.0 | `github.com/huggingface/transformers/blob/main/LICENSE` | ✅ Safe |
| **yastrebksv/TennisProject** | Fetched repo directly: **no LICENSE file**; 205★, 10 commits, Python; uses TrackNet + CatBoost + TennisCourtDetector | `github.com/yastrebksv/TennisProject` (repo sidebar shows no licence) | ⚠️ Reference only |
| **yastrebksv/TrackNet** | No LICENSE file; unofficial PyTorch TrackNet impl with pretrained weights | `github.com/yastrebksv/TrackNet` | ⚠️ Reference only |
| **yastrebksv/TennisCourtDetector** | No LICENSE file; 14-keypoint court model + homography | `github.com/yastrebksv/TennisCourtDetector` | ⚠️ Reference only |
| **ArtLabss/tennis-tracking** | "Monocular HawkEye"; TrackNet + ResNet50 + bounce classifier; last meaningful update early 2024; licence to verify | `github.com/ArtLabss/tennis-tracking` | ⚠️ Reference only; verify licence |
| **SportAI API** | Commercial B2B, quote-based pricing; vendor claims ~98% precision vs Hawk-Eye from 30fps/1080p phone video (not independently benchmarked) | sportai.com (confirm current API + pricing directly) | 💲 Paid, optional Phase 1+ |
| **Course-keypoints dataset (re-upload)** | 8,841 images, 14 annotated points each; mirror of the TennisCourtDetector dataset | `huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset` | Verify dataset licence before training on it |

**Accuracy-claim caveat:** All vendor accuracy figures in this PRD (SportAI's ~98%, and any SwingVision line-call figures referenced in the market analysis) are **company marketing claims, not independent benchmarks.** Treat them as directional and validate against your own footage before relying on them.

**Companion file:** `tennis-ai-coach-architecture.svg` — the validated system architecture diagram referenced in the Tech Architecture section.
