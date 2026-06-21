import pytest
from tennis_cv.segmentation import dominant_wrist_index, detect_serve_window
from tennis_cv.types import (
    Landmark, PoseFrame, NUM_LANDMARKS, L_WRIST, R_WRIST, NoPersonDetected,
)


def _frame(index, wrist_idx, wrist_y, detected=True):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    lms[wrist_idx] = Landmark(0.5, wrist_y, 0.0, 1.0)
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=detected)


def test_dominant_wrist_is_the_higher_reaching_one():
    # Right wrist reaches y=0.1 (higher); left stays at 0.5
    frames = [_frame(i, R_WRIST, 0.5 - 0.04 * i) for i in range(10)]
    assert dominant_wrist_index(frames) == R_WRIST


def test_contact_is_highest_point_and_start_is_lowest_before_it():
    # Wrist y dips to a max (lowest point) at frame 2, then rises to min at frame 6
    ys = [0.5, 0.6, 0.7, 0.5, 0.3, 0.15, 0.1, 0.2]
    frames = [_frame(i, R_WRIST, y) for i, y in enumerate(ys)]
    window = detect_serve_window(frames)
    assert window.contact_index == 6   # min y (highest)
    assert window.start_index == 2     # max y at/before contact (lowest)
    assert window.end_index == 7


def test_too_few_detections_raises():
    frames = [_frame(i, R_WRIST, 0.5, detected=False) for i in range(5)]
    with pytest.raises(NoPersonDetected):
        detect_serve_window(frames)
