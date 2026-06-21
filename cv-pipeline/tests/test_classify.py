from tennis_cv.classify import classify_stroke
from tennis_cv.types import (
    Landmark, PoseFrame, StrokeWindow, StrokeType, NUM_LANDMARKS, NOSE, R_WRIST,
)


def _frame(index, nose_y, wrist_y):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    lms[NOSE] = Landmark(0.5, nose_y, 0.0, 1.0)
    lms[R_WRIST] = Landmark(0.5, wrist_y, 0.0, 1.0)
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=True)


def test_wrist_above_head_at_contact_is_serve():
    frames = [_frame(0, nose_y=0.3, wrist_y=0.1)]  # wrist well above nose
    window = StrokeWindow(start_index=0, contact_index=0, end_index=0)
    assert classify_stroke(frames, window) == StrokeType.SERVE


def test_wrist_below_head_at_contact_is_unknown():
    frames = [_frame(0, nose_y=0.3, wrist_y=0.5)]  # wrist below nose
    window = StrokeWindow(start_index=0, contact_index=0, end_index=0)
    assert classify_stroke(frames, window) == StrokeType.UNKNOWN
