import math
import pytest
from tennis_cv.types import (
    Landmark, PoseFrame, StrokeType, angle, L_WRIST, NUM_LANDMARKS,
    AnalysisError, NotAServe,
)


def _lm(x, y):
    return Landmark(x=x, y=y, z=0.0, visibility=1.0)


def test_angle_right_angle():
    # b at origin, a straight up, c straight right -> 90 degrees
    a, b, c = _lm(0, 1), _lm(0, 0), _lm(1, 0)
    assert angle(a, b, c) == pytest.approx(90.0, abs=1e-6)


def test_angle_straight_line():
    a, b, c = _lm(-1, 0), _lm(0, 0), _lm(1, 0)
    assert angle(a, b, c) == pytest.approx(180.0, abs=1e-6)


def test_poseframe_lm_accessor():
    lms = [_lm(i, i) for i in range(NUM_LANDMARKS)]
    f = PoseFrame(index=0, timestamp=0.0, landmarks=lms, detected=True)
    assert f.lm(L_WRIST).x == L_WRIST


def test_stroketype_serialises_to_string():
    assert StrokeType.SERVE == "serve"


def test_error_hierarchy():
    assert issubclass(NotAServe, AnalysisError)
