# Drill Library + Deterministic Coaching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make serve coaching trustworthy by selecting drills deterministically from a vetted library and demoting the LLM to an optional "voice" pass that cannot change any fact.

**Architecture:** A new `drills.py` holds the vetted fault→drill library. `coaching.py` is refactored so `assemble_coaching` builds a complete, deterministic `Coaching` baseline from the scored metrics; an optional `apply_voice` LLM pass rewrites only the prose and is accepted only if `is_faithful` confirms it left every issue, drill, and number intact, otherwise the baseline is returned. A `preview.py` CLI renders representative coaching so a human can sign off the library.

**Tech Stack:** Python 3.11+ (built/run on 3.12), Pydantic v2, the `openai` SDK (OpenAI-compatible client, already a dependency), pytest. No new dependencies.

## Global Constraints

- **Package/root:** package `tennis_cv` under `cv-pipeline/`. Run tests with the repo venv: `/Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest ...` from `cv-pipeline/`.
- **Spec:** `docs/superpowers/specs/2026-06-21-drill-library-coaching-design.md` is the source of truth.
- **Determinism:** `assemble_coaching` and `is_faithful` are pure, offline, and never raise on valid `StrokeScore` input. Only the optional voice pass touches the network.
- **Fault cutoff:** a metric is a fault when its `sub_score` is **strictly below** `ISSUE_SUBSCORE_CUTOFF = 80.0`; at or above is a strength.
- **LLM role:** the voice pass may reword only the `summary` and each issue's `why`/`how`. It must not add/remove/rename/reorder issues or drills, and must not introduce numbers absent from the baseline. Any violation → return the baseline. Default model `gemini-2.5-flash-lite` via the OpenAI-compatible client (unchanged from current `coaching.py`).
- **Drill content is a v1 draft** pending coach review — a top-of-file comment in `drills.py` must say so.
- **Signature compatibility:** `generate_coaching(stroke_type, metrics, score, *, client=None, model=DEFAULT_COACHING_MODEL, use_voice=True) -> Coaching` — the pipeline calls it as `generate_coaching(stroke_type, score.factors, score, client=coaching_client)`, which must keep working. `Drill` must remain importable from `tennis_cv.coaching`.
- **TDD:** failing test first, watch it fail, minimal implementation, watch it pass, commit. Frequent commits.

---

### Task 1: Vetted drill library

**Files:**
- Create: `cv-pipeline/src/tennis_cv/drills.py`
- Test: `cv-pipeline/tests/test_drills.py`

**Interfaces:**
- Produces:
  - `class Drill(BaseModel)` — `name: str`, `description: str`.
  - `class FaultEntry(BaseModel)` — `metric: str`, `title: str`, `why: str`, `how: str`, `drills: list[Drill]`.
  - `DRILL_LIBRARY: dict[str, FaultEntry]` — entries keyed by the metric names the metrics module emits: `"knee_flexion"`, `"contact_height"`, `"arm_extension"`.
  - `def fault_entry(metric: str) -> FaultEntry | None`.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_drills.py`:

```python
from tennis_cv.drills import Drill, FaultEntry, DRILL_LIBRARY, fault_entry

# These are exactly the metric names emitted by tennis_cv.metrics.compute_serve_metrics.
SERVE_METRICS = {"knee_flexion", "contact_height", "arm_extension"}


def test_library_covers_every_serve_metric():
    assert SERVE_METRICS <= set(DRILL_LIBRARY)


def test_every_entry_is_well_formed():
    for metric, entry in DRILL_LIBRARY.items():
        assert isinstance(entry, FaultEntry)
        assert entry.metric == metric
        assert entry.title.strip()
        assert entry.why.strip()
        assert entry.how.strip()
        assert len(entry.drills) >= 1
        for drill in entry.drills:
            assert isinstance(drill, Drill)
            assert drill.name.strip()
            assert drill.description.strip()


def test_fault_entry_lookup():
    assert fault_entry("knee_flexion").title == DRILL_LIBRARY["knee_flexion"].title
    assert fault_entry("not_a_metric") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest tests/test_drills.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.drills'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/drills.py`:

```python
"""Vetted serve drill library.

v1 DRAFT — pending coach review. Drills are drafted from established serve
fundamentals. Review and replace the content via `python -m tennis_cv.preview`
before relying on it with real users. Adding a stroke later = adding entries
here; no code change elsewhere.
"""
from __future__ import annotations

from pydantic import BaseModel


class Drill(BaseModel):
    name: str
    description: str


class FaultEntry(BaseModel):
    metric: str
    title: str
    why: str
    how: str
    drills: list[Drill]


DRILL_LIBRARY: dict[str, FaultEntry] = {
    "knee_flexion": FaultEntry(
        metric="knee_flexion",
        title="Shallow leg load",
        why=(
            "Without a deep knee bend you lose the leg drive that powers an "
            "explosive serve, so the racquet head is slower at contact."
        ),
        how=(
            "Bend your knees more in the trophy position, then drive up and "
            "through the ball as you swing."
        ),
        drills=[
            Drill(
                name="Pause-and-drive",
                description=(
                    "Load into the trophy position, pause for one second to feel "
                    "the knee bend, then explode straight up. Ten reps, no ball."
                ),
            ),
            Drill(
                name="Wall-sit toss",
                description=(
                    "Hold a ninety-degree wall sit for twenty seconds before each "
                    "shadow serve to groove the loaded-leg feeling."
                ),
            ),
        ],
    ),
    "contact_height": FaultEntry(
        metric="contact_height",
        title="Low contact point",
        why=(
            "Striking the ball below full reach lowers your margin over the net "
            "and cuts the downward angle you can hit, costing power and consistency."
        ),
        how=(
            "Reach up and strike the ball at the top of your extension; toss a "
            "little higher and let the ball drop into the top of the strike zone."
        ),
        drills=[
            Drill(
                name="Hit-the-fence-top",
                description=(
                    "Stand near a fence and serve so contact is above the fence "
                    "line, forcing full extension. Fifteen serves."
                ),
            ),
            Drill(
                name="Tall-toss reach",
                description=(
                    "Toss and freeze with your tossing arm fully extended "
                    "overhead; only swing once you feel the stretch upward."
                ),
            ),
        ],
    ),
    "arm_extension": FaultEntry(
        metric="arm_extension",
        title="Bent arm at contact",
        why=(
            "A bent hitting arm at contact shortens your lever and leaks "
            "racquet-head speed, reducing power and making contact inconsistent."
        ),
        how=(
            "Extend your hitting arm so it is nearly straight at contact, reaching "
            "up to the ball rather than pulling down early."
        ),
        drills=[
            Drill(
                name="Straight-arm serves",
                description=(
                    "Serve at half pace focusing only on a fully straight hitting "
                    "arm at contact; build up speed once it feels natural."
                ),
            ),
            Drill(
                name="Trophy-to-contact freeze",
                description=(
                    "Shadow-swing from the trophy position to contact and freeze "
                    "at full extension to check the arm is straight."
                ),
            ),
        ],
    ),
}


def fault_entry(metric: str) -> FaultEntry | None:
    return DRILL_LIBRARY.get(metric)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest tests/test_drills.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/shaunlee/Desktop/apps/tennis-app
git add cv-pipeline/src/tennis_cv/drills.py cv-pipeline/tests/test_drills.py
git commit -m "feat(cv): add vetted serve drill library"
```

---

### Task 2: Deterministic coaching + guarded voice pass

**Files:**
- Modify (replace contents): `cv-pipeline/src/tennis_cv/coaching.py`
- Modify (rewrite): `cv-pipeline/tests/test_coaching.py`
- Modify: `cv-pipeline/tests/test_pipeline.py` (update the fake coaching client + coaching assertions)

**Interfaces:**
- Consumes: `Drill`, `fault_entry` from `tennis_cv.drills`; `MetricResult`, `StrokeScore`, `StrokeType` from `tennis_cv.types`.
- Produces:
  - `DEFAULT_COACHING_MODEL = "gemini-2.5-flash-lite"`, `DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"`, `ISSUE_SUBSCORE_CUTOFF = 80.0`.
  - `class Issue(BaseModel)` — `title: str`, `why: str`, `how: str`, `priority: int`, `drills: list[Drill]`.
  - `class Coaching(BaseModel)` — `summary: str`, `issues: list[Issue]`.
  - `Drill` re-exported (so `from tennis_cv.coaching import Drill` still works).
  - `def assemble_coaching(stroke_type: StrokeType, score: StrokeScore) -> Coaching`.
  - `def build_voice_prompt(baseline: Coaching, score: StrokeScore) -> str`.
  - `def apply_voice(baseline: Coaching, score: StrokeScore, *, client, model: str) -> Coaching`.
  - `def is_faithful(baseline: Coaching, rewrite: Coaching) -> bool`.
  - `def generate_coaching(stroke_type, metrics, score, *, client=None, model=DEFAULT_COACHING_MODEL, use_voice=True) -> Coaching`.

- [ ] **Step 1: Write the failing tests (coaching)**

Replace the entire contents of `cv-pipeline/tests/test_coaching.py` with:

```python
from types import SimpleNamespace

from tennis_cv.coaching import (
    Coaching, Issue, Drill, ISSUE_SUBSCORE_CUTOFF,
    assemble_coaching, build_voice_prompt, is_faithful, generate_coaching,
)
from tennis_cv.types import MetricResult, StrokeScore, StrokeType


def _factor(name, sub_score):
    # value/unit/targets are irrelevant to assembly; only name + sub_score matter.
    return MetricResult(name, 0.0, "u", 0.0, 1.0, sub_score)


def _score(*subscores):
    """Build a StrokeScore from (name, sub_score) pairs; overall = rounded mean."""
    factors = [_factor(n, s) for n, s in subscores]
    overall = round(sum(s for _, s in subscores) / len(subscores)) if subscores else 0
    return StrokeScore(overall=overall, factors=factors)


def test_assemble_flags_below_cutoff_metrics_as_issues_ranked_worst_first():
    score = _score(("knee_flexion", 50), ("contact_height", 100), ("arm_extension", 20))
    coaching = assemble_coaching(StrokeType.SERVE, score)
    titles = [i.title for i in coaching.issues]
    # arm_extension (20) is worst -> priority 1; knee_flexion (50) -> priority 2.
    assert titles == ["Bent arm at contact", "Shallow leg load"]
    assert [i.priority for i in coaching.issues] == [1, 2]
    assert all(len(i.drills) >= 1 for i in coaching.issues)
    assert str(score.overall) in coaching.summary


def test_assemble_with_no_faults_has_no_issues_and_positive_summary():
    score = _score(("knee_flexion", 100), ("contact_height", 90), ("arm_extension", 85))
    coaching = assemble_coaching(StrokeType.SERVE, score)
    assert coaching.issues == []
    assert "no major" in coaching.summary.lower()


def test_cutoff_is_strict():
    # Exactly at the cutoff is a strength, not an issue.
    score = _score(("knee_flexion", ISSUE_SUBSCORE_CUTOFF))
    assert assemble_coaching(StrokeType.SERVE, score).issues == []


def _baseline():
    return assemble_coaching(StrokeType.SERVE, _score(("knee_flexion", 40)))


def test_is_faithful_accepts_prose_only_rewrite():
    base = _baseline()
    issue = base.issues[0]
    reworded = Issue(
        title=issue.title,
        why="A reworded but number-free explanation of the same point.",
        how="A friendlier cue, still number-free.",
        priority=issue.priority,
        drills=issue.drills,  # drills kept verbatim
    )
    # summary reuses only numbers already in the baseline summary.
    rewrite = Coaching(summary=base.summary, issues=[reworded])
    assert is_faithful(base, rewrite) is True


def test_is_faithful_rejects_dropped_issue():
    base = _baseline()
    assert is_faithful(base, Coaching(summary=base.summary, issues=[])) is False


def test_is_faithful_rejects_changed_drill():
    base = _baseline()
    issue = base.issues[0]
    tampered = Issue(
        title=issue.title, why=issue.why, how=issue.how, priority=issue.priority,
        drills=[Drill(name="Totally different drill", description="x")],
    )
    assert is_faithful(base, Coaching(summary=base.summary, issues=[tampered])) is False


def test_is_faithful_rejects_invented_number():
    base = _baseline()
    issue = base.issues[0]
    bad = Issue(
        title=issue.title, why="Improve by 35 percent.", how=issue.how,
        priority=issue.priority, drills=issue.drills,
    )
    assert is_faithful(base, Coaching(summary=base.summary, issues=[bad])) is False


def test_build_voice_prompt_mentions_constraints_and_drills():
    base = _baseline()
    prompt = build_voice_prompt(base, _score(("knee_flexion", 40)))
    assert "Pause-and-drive" in prompt          # a drill name from the baseline
    assert "summary" in prompt and "drills" in prompt


def _faithful_client(baseline):
    # Returns the baseline unchanged -> trivially faithful.
    def parse(**kwargs):
        message = SimpleNamespace(parsed=baseline)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def _unfaithful_client():
    def parse(**kwargs):
        bad = Coaching(summary="Improve 999 percent", issues=[])
        message = SimpleNamespace(parsed=bad)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def _raising_client():
    def parse(**kwargs):
        raise RuntimeError("network down")
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def test_generate_use_voice_false_returns_baseline():
    score = _score(("knee_flexion", 40))
    result = generate_coaching(StrokeType.SERVE, score.factors, score, use_voice=False)
    assert result == assemble_coaching(StrokeType.SERVE, score)


def test_generate_accepts_faithful_voice():
    score = _score(("knee_flexion", 40))
    baseline = assemble_coaching(StrokeType.SERVE, score)
    result = generate_coaching(
        StrokeType.SERVE, score.factors, score, client=_faithful_client(baseline),
    )
    assert result == baseline


def test_generate_rejects_unfaithful_voice_and_falls_back():
    score = _score(("knee_flexion", 40))
    result = generate_coaching(
        StrokeType.SERVE, score.factors, score, client=_unfaithful_client(),
    )
    assert result == assemble_coaching(StrokeType.SERVE, score)


def test_generate_falls_back_when_client_raises():
    score = _score(("knee_flexion", 40))
    result = generate_coaching(
        StrokeType.SERVE, score.factors, score, client=_raising_client(),
    )
    assert result == assemble_coaching(StrokeType.SERVE, score)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest tests/test_coaching.py -v`
Expected: FAIL with `ImportError` (e.g. cannot import `assemble_coaching` / `ISSUE_SUBSCORE_CUTOFF`).

- [ ] **Step 3: Replace `coaching.py`**

Replace the entire contents of `cv-pipeline/src/tennis_cv/coaching.py` with:

```python
from __future__ import annotations

import os
import re

from pydantic import BaseModel

from .drills import Drill, fault_entry
from .types import MetricResult, StrokeScore, StrokeType

# Coaching runs on any OpenAI-compatible chat endpoint, so the provider is a
# config choice, not a code change. Default to Gemini 2.5 Flash-Lite; swap to
# Qwen / DeepSeek / Claude / a gateway by overriding model, base URL, and key.
DEFAULT_COACHING_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# A metric is a fault when its sub-score is strictly below this; at/above is a strength.
ISSUE_SUBSCORE_CUTOFF = 80.0

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class Issue(BaseModel):
    title: str
    why: str
    how: str
    priority: int
    drills: list[Drill]


class Coaching(BaseModel):
    summary: str
    issues: list[Issue]


def _summary(stroke_type: StrokeType, score: StrokeScore, issues: list[Issue]) -> str:
    stroke = stroke_type.value
    if not issues:
        if score.factors:
            best = max(score.factors, key=lambda f: f.sub_score)
            return (
                f"Strong {stroke} — you scored {score.overall}/100 with no major "
                f"technique faults. Your {best.name.replace('_', ' ')} looks "
                f"especially solid. Keep it up."
            )
        return f"Your {stroke} scored {score.overall}/100."
    top = issues[0]
    return (
        f"Your {stroke} scored {score.overall}/100. The biggest thing to work on "
        f"is {top.title.lower()} — see the drills below."
    )


def assemble_coaching(stroke_type: StrokeType, score: StrokeScore) -> Coaching:
    faults = [f for f in score.factors if f.sub_score < ISSUE_SUBSCORE_CUTOFF]
    faults.sort(key=lambda f: f.sub_score)  # worst (lowest sub-score) first
    issues: list[Issue] = []
    for priority, factor in enumerate(faults, start=1):
        entry = fault_entry(factor.name)
        if entry is None:
            continue  # unknown metric -> no drill content; covered by test_drills
        issues.append(
            Issue(
                title=entry.title,
                why=entry.why,
                how=entry.how,
                priority=priority,
                drills=entry.drills,
            )
        )
    return Coaching(summary=_summary(stroke_type, score, issues), issues=issues)


def build_voice_prompt(baseline: Coaching, score: StrokeScore) -> str:
    import json

    payload = json.dumps(baseline.model_dump(), indent=2)
    return (
        "You are a warm, encouraging tennis coach. Rewrite the coaching feedback "
        "below in a friendly, natural voice.\n\n"
        "STRICT RULES:\n"
        "- Keep exactly the same issues (same 'title' values) and exactly the same "
        "'drills' (same name and description, verbatim). Do not add, remove, rename, "
        "or reorder issues or drills.\n"
        "- You may only reword the top-level 'summary' and each issue's 'why' and "
        "'how'.\n"
        "- Do not introduce any number that is not already present.\n"
        f"- The overall score is {score.overall}/100.\n\n"
        f"Feedback to rewrite (JSON):\n{payload}"
    )


def apply_voice(baseline: Coaching, score: StrokeScore, *, client, model: str) -> Coaching:
    completion = client.chat.completions.parse(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": build_voice_prompt(baseline, score)}],
        response_format=Coaching,
    )
    return completion.choices[0].message.parsed


def _numbers(*texts: str) -> set[str]:
    found: set[str] = set()
    for text in texts:
        found.update(_NUMBER_RE.findall(text))
    return found


def is_faithful(baseline: Coaching, rewrite: Coaching) -> bool:
    base_by_title = {i.title: i for i in baseline.issues}
    new_by_title = {i.title: i for i in rewrite.issues}
    if set(base_by_title) != set(new_by_title):
        return False
    if len(baseline.issues) != len(rewrite.issues):
        return False
    for title, base_issue in base_by_title.items():
        new_issue = new_by_title[title]
        if base_issue.drills != new_issue.drills:
            return False  # drills must be verbatim
        if not _numbers(new_issue.why, new_issue.how) <= _numbers(base_issue.why, base_issue.how):
            return False  # no invented numbers in the prose
    if not _numbers(rewrite.summary) <= _numbers(baseline.summary):
        return False
    return True


def generate_coaching(
    stroke_type: StrokeType,
    metrics: list[MetricResult],
    score: StrokeScore,
    *,
    client=None,
    model: str = DEFAULT_COACHING_MODEL,
    use_voice: bool = True,
) -> Coaching:
    # `metrics` is accepted for call-site compatibility; assembly reads score.factors.
    baseline = assemble_coaching(stroke_type, score)
    if not use_voice:
        return baseline
    try:
        if client is None:
            import openai

            client = openai.OpenAI(
                base_url=os.environ.get("COACHING_BASE_URL", DEFAULT_BASE_URL),
                api_key=(
                    os.environ.get("COACHING_API_KEY")
                    or os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                ),
            )
        rewrite = apply_voice(baseline, score, client=client, model=model)
    except Exception:
        return baseline
    return rewrite if is_faithful(baseline, rewrite) else baseline
```

- [ ] **Step 4: Run coaching tests to verify they pass**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest tests/test_coaching.py -v`
Expected: PASS (all coaching tests pass).

- [ ] **Step 5: Update the pipeline test for the new coaching shape**

In `cv-pipeline/tests/test_pipeline.py`, replace the `_fake_coaching_client` helper (the LLM is now a voice pass that the test forces to fall back to the deterministic baseline) with:

```python
def _fake_coaching_client():
    # Force the deterministic baseline: the voice pass raises, so generate_coaching
    # falls back. Keeps the pipeline test offline and independent of coaching prose.
    def parse(**kwargs):
        raise RuntimeError("voice disabled in test")

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )
```

Then, in `test_analyze_serve_end_to_end`, replace the coaching assertion line:

```python
    assert result.coaching.summary.startswith("Good extension")
```

with:

```python
    assert isinstance(result.coaching.summary, str) and result.coaching.summary
    assert isinstance(result.coaching.issues, list)
```

If `test_pipeline.py` imports `Issue` and/or `Drill` from `tennis_cv.coaching` only for the old fake, leave the imports — they still resolve. (Do not change anything else in the file.)

- [ ] **Step 6: Run the full suite to verify no regressions**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest -v`
Expected: PASS (all tests across the package pass).

- [ ] **Step 7: Commit**

```bash
cd /Users/shaunlee/Desktop/apps/tennis-app
git add cv-pipeline/src/tennis_cv/coaching.py cv-pipeline/tests/test_coaching.py cv-pipeline/tests/test_pipeline.py
git commit -m "feat(cv): deterministic coaching from library with guarded voice pass"
```

---

### Task 3: Preview CLI for library sign-off

**Files:**
- Create: `cv-pipeline/src/tennis_cv/preview.py`
- Test: `cv-pipeline/tests/test_preview.py`

**Interfaces:**
- Consumes: `assemble_coaching`, `generate_coaching`, `Coaching` from `tennis_cv.coaching`; `MetricResult`, `StrokeScore`, `StrokeType` from `tennis_cv.types`.
- Produces:
  - `REPRESENTATIVE_PROFILES: list[tuple[str, StrokeScore]]` — named serve profiles covering shallow knee only, low contact only, bent arm only, all-good, and multi-fault.
  - `def render_profile(name: str, score: StrokeScore, *, use_voice: bool, client=None) -> str`.
  - `def main(argv: list[str] | None = None) -> int` — prints every profile; deterministic by default, `--voice` includes the LLM pass.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_preview.py`:

```python
from tennis_cv.preview import REPRESENTATIVE_PROFILES, render_profile, main


def test_profiles_cover_the_expected_cases():
    names = [name for name, _ in REPRESENTATIVE_PROFILES]
    assert "all-good" in names
    assert any("multi" in n for n in names)
    assert len(REPRESENTATIVE_PROFILES) >= 5


def test_render_profile_deterministic_contains_drill_for_a_fault():
    # Find the shallow-knee profile and confirm its rendered text names a knee drill.
    name, score = next(p for p in REPRESENTATIVE_PROFILES if p[0] == "shallow-knee")
    text = render_profile(name, score, use_voice=False)
    assert "Pause-and-drive" in text
    assert "Shallow leg load" in text


def test_main_renders_all_profiles(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    for name, _ in REPRESENTATIVE_PROFILES:
        assert name in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest tests/test_preview.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.preview'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/preview.py`:

```python
from __future__ import annotations

import argparse

from .coaching import Coaching, assemble_coaching, generate_coaching
from .types import MetricResult, StrokeScore, StrokeType

# Metric sub-scores below 80 are faults (ISSUE_SUBSCORE_CUTOFF). value/unit/targets
# are irrelevant to coaching assembly, so use placeholders.
_METRICS = ("knee_flexion", "contact_height", "arm_extension")


def _profile(**subscores: float) -> StrokeScore:
    factors = [
        MetricResult(name, 0.0, "u", 0.0, 1.0, subscores.get(name, 100.0))
        for name in _METRICS
    ]
    overall = round(sum(f.sub_score for f in factors) / len(factors))
    return StrokeScore(overall=overall, factors=factors)


REPRESENTATIVE_PROFILES: list[tuple[str, StrokeScore]] = [
    ("shallow-knee", _profile(knee_flexion=40.0)),
    ("low-contact", _profile(contact_height=35.0)),
    ("bent-arm", _profile(arm_extension=30.0)),
    ("all-good", _profile()),
    ("multi-fault", _profile(knee_flexion=45.0, contact_height=50.0, arm_extension=25.0)),
]


def _format(coaching: Coaching) -> str:
    lines = [f"  {coaching.summary}"]
    for issue in coaching.issues:
        lines.append(f"  [{issue.priority}] {issue.title}")
        lines.append(f"      why: {issue.why}")
        lines.append(f"      how: {issue.how}")
        for drill in issue.drills:
            lines.append(f"      drill — {drill.name}: {drill.description}")
    return "\n".join(lines)


def render_profile(name: str, score: StrokeScore, *, use_voice: bool, client=None) -> str:
    if use_voice:
        coaching = generate_coaching(StrokeType.SERVE, score.factors, score, client=client)
    else:
        coaching = assemble_coaching(StrokeType.SERVE, score)
    return f"=== {name} (score {score.overall}/100) ===\n{_format(coaching)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview serve coaching for sample profiles.")
    parser.add_argument(
        "--voice", action="store_true",
        help="Include the LLM voice pass (needs an API key); off = deterministic baseline.",
    )
    args = parser.parse_args(argv)
    for name, score in REPRESENTATIVE_PROFILES:
        print(render_profile(name, score, use_voice=args.voice))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest tests/test_preview.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m pytest -q`
Expected: PASS (all tests pass).

- [ ] **Step 6: Commit**

```bash
cd /Users/shaunlee/Desktop/apps/tennis-app
git add cv-pipeline/src/tennis_cv/preview.py cv-pipeline/tests/test_preview.py
git commit -m "feat(cv): add coaching preview CLI for drill-library sign-off"
```

---

## Manual sign-off (after Task 3)

For the human library review the spec calls for:

1. `cd cv-pipeline && /Users/shaunlee/Desktop/apps/tennis-app/.venv/bin/python -m tennis_cv.preview`
2. Read each profile's deterministic coaching; confirm the issue, why/how, and drills are correct, safe, and well-phrased for that fault. Edit `drills.py` and re-run until you (or a coach) would sign off.
3. Optionally `... -m tennis_cv.preview --voice` with `GEMINI_API_KEY` set to see the warmed-up phrasing and confirm it stays faithful.

## Plan self-review notes

- **Spec coverage:** drill library (Task 1); deterministic `assemble_coaching` + cutoff/strength logic + priority (Task 2); optional `apply_voice` + `is_faithful` guard + graceful fallback (Task 2); preview CLI (Task 3); all error/edge cases (missing entry skipped + asserted in Task 1; no-faults summary; LLM disabled/error/unfaithful → baseline). Migration: `Drill` re-exported, pipeline signature unchanged, pipeline test updated.
- **Determinism preserved:** `assemble_coaching`/`is_faithful` are pure; the only network call is the guarded voice pass, and every failure mode returns the baseline.
- **Type consistency:** `Issue` carries `drills`; `Coaching` is `summary + issues`; `generate_coaching` keeps the `(stroke_type, metrics, score, *, client, model, use_voice)` shape the pipeline calls; `is_faithful` matches issues by `title` (the voice prompt leaves `title`/`drills` verbatim) and compares `drills` by full equality.
- **No placeholders:** every step has complete code and exact commands.
