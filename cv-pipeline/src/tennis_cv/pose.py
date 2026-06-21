from __future__ import annotations

from typing import Protocol

from .types import Landmark, NUM_LANDMARKS, PoseFrame, RawFrame


def landmarks_from_mp(result, num_landmarks: int = NUM_LANDMARKS) -> tuple[list[Landmark], bool]:
    if not getattr(result, "pose_landmarks", None):
        zero = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(num_landmarks)]
        return zero, False
    out = [
        Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
        for lm in result.pose_landmarks.landmark
    ]
    return out, True


class PoseEstimator(Protocol):
    def estimate_all(self, raw_frames: list[RawFrame]) -> list[PoseFrame]:
        ...


class MediaPipePoseEstimator:
    """Runs MediaPipe legacy Pose over decoded frames."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence

    def estimate_all(self, raw_frames: list[RawFrame]) -> list[PoseFrame]:
        import cv2
        import mediapipe as mp

        pose_frames: list[PoseFrame] = []
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_detection_confidence,
        ) as pose:
            for raw in raw_frames:
                rgb = cv2.cvtColor(raw.image, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                landmarks, detected = landmarks_from_mp(result)
                pose_frames.append(
                    PoseFrame(
                        index=raw.index,
                        timestamp=raw.timestamp,
                        landmarks=landmarks,
                        detected=detected,
                    )
                )
        return pose_frames
