from __future__ import annotations

from .segmentation import dominant_wrist_index
from .types import NOSE, PoseFrame, StrokeType, StrokeWindow

SERVE_WRIST_ABOVE_NOSE_MARGIN = 0.05


def classify_stroke(frames: list[PoseFrame], window: StrokeWindow) -> StrokeType:
    by_index = {f.index: f for f in frames}
    contact = by_index[window.contact_index]
    wrist = dominant_wrist_index(frames)
    wrist_y = contact.lm(wrist).y
    nose_y = contact.lm(NOSE).y
    if wrist_y < nose_y - SERVE_WRIST_ABOVE_NOSE_MARGIN:
        return StrokeType.SERVE
    return StrokeType.UNKNOWN
