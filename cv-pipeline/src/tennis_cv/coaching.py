from __future__ import annotations

import os

from pydantic import BaseModel

from .types import MetricResult, StrokeScore, StrokeType

# Coaching runs on any OpenAI-compatible chat endpoint, so the provider is a
# config choice, not a code change. Default to Gemini 2.5 Flash-Lite (cheapest
# tier with strong structured output); swap to Qwen / DeepSeek / Claude / a
# gateway by overriding the model, base URL, and key via env vars.
DEFAULT_COACHING_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class Issue(BaseModel):
    title: str
    why: str
    how: str
    priority: int


class Drill(BaseModel):
    name: str
    description: str
    addresses: str


class Coaching(BaseModel):
    summary: str
    issues: list[Issue]
    drills: list[Drill]


def build_prompt(stroke_type: StrokeType, metrics: list[MetricResult], score: StrokeScore) -> str:
    lines = [
        f"You are a tennis coach analysing a {stroke_type.value}.",
        f"Overall technique score: {score.overall}/100.",
        "",
        "Measured metrics (value, unit, target range, sub-score 0-100):",
    ]
    for m in metrics:
        lines.append(
            f"- {m.name}: {m.value:.1f} {m.unit} "
            f"(target {m.target_low:.1f}-{m.target_high:.1f}, sub-score {m.sub_score:.0f})"
        )
    lines += [
        "",
        "Ground every observation in these measured numbers. Do not invent any "
        "measurement that is not listed above. Rank issues by impact (priority 1 "
        "= most important). For each issue give a specific corrective drill.",
    ]
    return "\n".join(lines)


def generate_coaching(
    stroke_type: StrokeType,
    metrics: list[MetricResult],
    score: StrokeScore,
    *,
    client=None,
    model: str = DEFAULT_COACHING_MODEL,
) -> Coaching:
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
    prompt = build_prompt(stroke_type, metrics, score)
    completion = client.chat.completions.parse(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        response_format=Coaching,
    )
    return completion.choices[0].message.parsed
