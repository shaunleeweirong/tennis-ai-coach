import numpy as np
import cv2
import pytest
from types import SimpleNamespace
from tennis_cv.pipeline import analyze
from tennis_cv.coaching import Coaching, Issue, Drill
from tennis_cv.types import (
    Landmark, PoseFrame, RawFrame, NUM_LANDMARKS, StrokeType, NotAServe,
    NOSE, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE,
)


def _write_video(path, n_frames=6, w=64, h=48, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def _serve_pose(index, wrist_y):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    lms[NOSE] = Landmark(0.5, 0.4, 0.0, 1.0)
    lms[R_SHOULDER] = Landmark(0.5, 0.45, 0.0, 1.0)
    lms[R_ELBOW] = Landmark(0.5, 0.3, 0.0, 1.0)
    lms[R_WRIST] = Landmark(0.5, wrist_y, 0.0, 1.0)
    lms[R_HIP] = Landmark(0.5, 0.6, 0.0, 1.0)
    lms[R_KNEE] = Landmark(0.5, 0.75, 0.0, 1.0)
    lms[R_ANKLE] = Landmark(0.5, 0.9, 0.0, 1.0)
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=True)


class _FakeEstimator:
    def __init__(self, poses):
        self._poses = poses

    def estimate_all(self, raw_frames: list[RawFrame]) -> list[PoseFrame]:
        return self._poses


def _fake_coaching_client():
    coaching = Coaching(
        summary="Good extension; load the legs more.",
        issues=[Issue(title="Shallow bend", why="x", how="y", priority=1)],
        drills=[Drill(name="d", description="e", addresses="knee_flexion")],
    )
    return SimpleNamespace(
        messages=SimpleNamespace(parse=lambda **kw: SimpleNamespace(parsed_output=coaching))
    )


def test_analyze_serve_end_to_end(tmp_path):
    vid = tmp_path / "serve.mp4"
    _write_video(vid, n_frames=6)
    # Wrist rises to a clear peak (min y) at frame 3 -> above the nose (0.4).
    wrist_ys = [0.7, 0.6, 0.4, 0.1, 0.3, 0.5]
    poses = [_serve_pose(i, y) for i, y in enumerate(wrist_ys)]

    result = analyze(
        str(vid),
        str(tmp_path / "out"),
        estimator=_FakeEstimator(poses),
        coaching_client=_fake_coaching_client(),
    )

    assert result.stroke_type == StrokeType.SERVE
    assert 0 <= result.score.overall <= 100
    assert len(result.metrics) == 3
    assert len(result.overlay_paths) == 3
    assert result.coaching.summary.startswith("Good extension")
    assert result.duration == pytest.approx(5 / 30, abs=1e-3)


def test_analyze_rejects_non_serve(tmp_path):
    vid = tmp_path / "not_serve.mp4"
    _write_video(vid, n_frames=6)
    # Wrist never rises above the nose -> classifier returns UNKNOWN.
    poses = [_serve_pose(i, 0.6) for i in range(6)]
    with pytest.raises(NotAServe):
        analyze(
            str(vid),
            str(tmp_path / "out"),
            estimator=_FakeEstimator(poses),
            coaching_client=_fake_coaching_client(),
        )
