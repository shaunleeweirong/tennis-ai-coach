from __future__ import annotations

from .types import (
    L_WRIST, R_WRIST, NoPersonDetected, PoseFrame, StrokeWindow,
)


def dominant_wrist_index(frames: list[PoseFrame]) -> int:
    detected = [f for f in frames if f.detected]
    left_min = min((f.lm(L_WRIST).y for f in detected), default=1.0)
    right_min = min((f.lm(R_WRIST).y for f in detected), default=1.0)
    return R_WRIST if right_min <= left_min else L_WRIST


def detect_serve_window(frames: list[PoseFrame]) -> StrokeWindow:
    detected = [f for f in frames if f.detected]
    if len(detected) < 3:
        raise NoPersonDetected("Not enough frames with a detected person.")
    wrist = dominant_wrist_index(detected)

    # Contact = globally highest wrist position (minimum y).
    contact = min(detected, key=lambda f: f.lm(wrist).y)
    contact_index = contact.index

    # Start = lowest wrist position (maximum y) at or before contact.
    before = [f for f in detected if f.index <= contact_index]
    start = max(before, key=lambda f: f.lm(wrist).y)

    end_index = detected[-1].index
    return StrokeWindow(
        start_index=start.index,
        contact_index=contact_index,
        end_index=end_index,
    )
