import pytest
from tennis_cv.metrics import compute_serve_metrics
from tennis_cv.types import (
    Landmark, PoseFrame, StrokeWindow, NUM_LANDMARKS,
    NOSE, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE,
)


def _serve_frame(index, *, knee_angle_pts, shoulder, elbow, wrist, hip):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    hip_pt, knee_pt, ankle_pt = knee_angle_pts
    lms[R_HIP] = Landmark(*hip_pt, 0.0, 1.0)
    lms[R_KNEE] = Landmark(*knee_pt, 0.0, 1.0)
    lms[R_ANKLE] = Landmark(*ankle_pt, 0.0, 1.0)
    lms[R_SHOULDER] = Landmark(*shoulder, 0.0, 1.0)
    lms[R_ELBOW] = Landmark(*elbow, 0.0, 1.0)
    lms[R_WRIST] = Landmark(*wrist, 0.0, 1.0)
    # Dominant-wrist detection needs the right wrist to be the higher one:
    lms[NOSE] = Landmark(0.5, 0.4, 0.0, 1.0)
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=True)


def test_computes_three_named_metrics_with_expected_values():
    # Contact frame: arm straight up (extension ~180), wrist above shoulder.
    # hip_y=0.6, shoulder_y=0.4, wrist_y=0.1 -> contact_height = (0.6-0.1)/(0.6-0.4) = 2.5
    contact = _serve_frame(
        0,
        knee_angle_pts=((0.5, 0.6), (0.5, 0.75), (0.5, 0.9)),  # straight leg ~180
        shoulder=(0.5, 0.4),
        elbow=(0.5, 0.25),
        wrist=(0.5, 0.1),
        hip=(0.5, 0.6),
    )
    # Loading frame: deep knee bend ~90 degrees (ankle directly below knee, hip forward)
    loading = _serve_frame(
        1,
        knee_angle_pts=((0.7, 0.7), (0.5, 0.7), (0.5, 0.9)),
        shoulder=(0.5, 0.4),
        elbow=(0.5, 0.5),
        wrist=(0.5, 0.6),
        hip=(0.5, 0.6),
    )
    window = StrokeWindow(start_index=1, contact_index=0, end_index=1)
    metrics = {m.name: m for m in compute_serve_metrics([contact, loading], window)}

    assert set(metrics) == {"knee_flexion", "contact_height", "arm_extension"}
    assert metrics["contact_height"].value == pytest.approx(2.5, abs=1e-6)
    assert metrics["arm_extension"].value == pytest.approx(180.0, abs=1.0)
    # Deepest knee bend across window is the ~90-degree loading frame.
    assert metrics["knee_flexion"].value == pytest.approx(90.0, abs=5.0)
