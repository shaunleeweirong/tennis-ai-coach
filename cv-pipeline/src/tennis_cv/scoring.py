from __future__ import annotations

from dataclasses import replace

from .types import MetricResult, StrokeScore


def sub_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 100.0
    width = max(high - low, 1e-9)
    distance = (low - value) if value < low else (value - high)
    return max(0.0, 100.0 * (1 - distance / width))


def score_serve(metrics: list[MetricResult]) -> StrokeScore:
    factors = [
        replace(m, sub_score=sub_score(m.value, m.target_low, m.target_high))
        for m in metrics
    ]
    mean = sum(f.sub_score for f in factors) / len(factors) if factors else 0.0
    return StrokeScore(overall=round(mean), factors=factors)
