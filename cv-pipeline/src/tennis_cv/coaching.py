from __future__ import annotations

from pydantic import BaseModel

from .types import MetricResult, StrokeScore, StrokeType


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
    model: str = "claude-opus-4-8",
) -> Coaching:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    prompt = build_prompt(stroke_type, metrics, score)
    response = client.messages.parse(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        output_format=Coaching,
    )
    return response.parsed_output
