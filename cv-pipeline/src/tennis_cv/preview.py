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
