from __future__ import annotations

import os

import cv2

from .types import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    PoseFrame, StrokeWindow,
)

SKELETON_EDGES: list[tuple[int, int]] = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


def _draw_skeleton(image, pose: PoseFrame) -> None:
    h, w = image.shape[:2]
    for a, b in SKELETON_EDGES:
        pa, pb = pose.lm(a), pose.lm(b)
        cv2.line(
            image,
            (int(pa.x * w), int(pa.y * h)),
            (int(pb.x * w), int(pb.y * h)),
            (0, 255, 0),
            2,
        )


def render_overlays(
    video_path: str,
    pose_frames: list[PoseFrame],
    window: StrokeWindow,
    out_dir: str,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    by_index = {f.index: f for f in pose_frames}
    key_indices = [window.start_index, window.contact_index, window.end_index]

    cap = cv2.VideoCapture(video_path)
    paths: list[str] = []
    for idx in key_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, image = cap.read()
        if not ok:
            continue
        pose = by_index.get(idx)
        if pose is not None and pose.detected:
            _draw_skeleton(image, pose)
        path = os.path.join(out_dir, f"frame_{idx}.png")
        cv2.imwrite(path, image)
        paths.append(path)
    cap.release()
    return paths
