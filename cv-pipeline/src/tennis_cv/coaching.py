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
