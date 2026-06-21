from tennis_cv.pose import landmarks_from_mp
from tennis_cv.types import NUM_LANDMARKS


class _FakeLm:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeLandmarkList:
    def __init__(self, lms):
        self.landmark = lms


class _FakeResult:
    def __init__(self, lms):
        self.pose_landmarks = _FakeLandmarkList(lms) if lms else None


def test_converts_landmarks_when_detected():
    lms = [_FakeLm(0.1 * i, 0.2 * i, 0.0, 0.9) for i in range(NUM_LANDMARKS)]
    out, detected = landmarks_from_mp(_FakeResult(lms))
    assert detected is True
    assert len(out) == NUM_LANDMARKS
    assert out[5].x == 0.5
    assert out[5].visibility == 0.9


def test_missing_pose_returns_not_detected():
    out, detected = landmarks_from_mp(_FakeResult(None))
    assert detected is False
    assert len(out) == NUM_LANDMARKS
    assert all(lm.visibility == 0.0 for lm in out)
