from __future__ import annotations

from .segmentation import dominant_wrist_index
from .types import (
    L_WRIST, L_SHOULDER, L_ELBOW, L_HIP, L_KNEE, L_ANKLE,
    R_WRIST, R_SHOULDER, R_ELBOW, R_HIP, R_KNEE, R_ANKLE,
    MetricResult, PoseFrame, StrokeWindow, angle,
)

KNEE_FLEXION_TARGET = (110.0, 140.0)
CONTACT_HEIGHT_TARGET = (1.5, 2.5)
ARM_EXTENSION_TARGET = (160.0, 180.0)


def _dominant_side(frames: list[PoseFrame]) -> dict[str, int]:
    if dominant_wrist_index(frames) == L_WRIST:
        return {"shoulder": L_SHOULDER, "elbow": L_ELBOW, "wrist": L_WRIST,
                "hip": L_HIP, "knee": L_KNEE, "ankle": L_ANKLE}
    return {"shoulder": R_SHOULDER, "elbow": R_ELBOW, "wrist": R_WRIST,
            "hip": R_HIP, "knee": R_KNEE, "ankle": R_ANKLE}


def compute_serve_metrics(frames: list[PoseFrame], window: StrokeWindow) -> list[MetricResult]:
    side = _dominant_side(frames)
    by_index = {f.index: f for f in frames}
    contact = by_index[window.contact_index]

    in_window = [
        f for f in frames
        if window.start_index <= f.index <= window.end_index and f.detected
    ] or [contact]
    knee_flexion = min(
        angle(f.lm(side["hip"]), f.lm(side["knee"]), f.lm(side["ankle"]))
        for f in in_window
    )

    hip_y = contact.lm(side["hip"]).y
    shoulder_y = contact.lm(side["shoulder"]).y
    wrist_y = contact.lm(side["wrist"]).y
    denom = hip_y - shoulder_y
    contact_height = (hip_y - wrist_y) / denom if denom != 0 else 0.0

    arm_extension = angle(
        contact.lm(side["shoulder"]), contact.lm(side["elbow"]), contact.lm(side["wrist"])
    )

    return [
        MetricResult("knee_flexion", knee_flexion, "deg", *KNEE_FLEXION_TARGET, 0.0),
        MetricResult("contact_height", contact_height, "ratio", *CONTACT_HEIGHT_TARGET, 0.0),
        MetricResult("arm_extension", arm_extension, "deg", *ARM_EXTENSION_TARGET, 0.0),
    ]
