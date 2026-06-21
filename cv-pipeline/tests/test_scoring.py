import pytest
from tennis_cv.scoring import sub_score, score_serve
from tennis_cv.types import MetricResult


def test_sub_score_inside_range_is_100():
    assert sub_score(125, 110, 140) == 100.0


def test_sub_score_one_width_outside_is_zero():
    # width = 30; value 30 below low -> exactly one width outside -> 0
    assert sub_score(80, 110, 140) == pytest.approx(0.0, abs=1e-6)


def test_sub_score_half_width_outside_is_50():
    assert sub_score(95, 110, 140) == pytest.approx(50.0, abs=1e-6)


def test_score_serve_fills_subscores_and_averages():
    metrics = [
        MetricResult("knee_flexion", 125, "deg", 110, 140, 0.0),   # inside -> 100
        MetricResult("contact_height", 2.0, "ratio", 1.5, 2.5, 0.0),  # inside -> 100
        MetricResult("arm_extension", 170, "deg", 160, 180, 0.0),  # inside -> 100
    ]
    result = score_serve(metrics)
    assert result.overall == 100
    assert all(f.sub_score == 100.0 for f in result.factors)
    # Source metrics are not mutated.
    assert metrics[0].sub_score == 0.0
