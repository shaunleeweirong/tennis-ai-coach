from __future__ import annotations

import cv2

from .types import AnalysisError, RawFrame


def read_frames(video_path: str, max_frames: int | None = None) -> list[RawFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    frames: list[RawFrame] = []
    index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(RawFrame(index=index, timestamp=index / fps, image=image))
        index += 1
        if max_frames is not None and index >= max_frames:
            break
    cap.release()
    if not frames:
        raise AnalysisError(f"Video has no readable frames: {video_path}")
    return frames
