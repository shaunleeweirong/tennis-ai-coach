from __future__ import annotations

from dataclasses import dataclass

from .classify import classify_stroke
from .coaching import Coaching, generate_coaching
from .frames import read_frames
from .metrics import compute_serve_metrics
from .overlay import render_overlays
from .pose import MediaPipePoseEstimator, PoseEstimator
from .scoring import score_serve
from .segmentation import detect_serve_window
from .types import (
    MetricResult, NotAServe, StrokeScore, StrokeType, StrokeWindow,
)


@dataclass
class AnalysisResult:
    stroke_type: StrokeType
    score: StrokeScore
    metrics: list[MetricResult]
    overlay_paths: list[str]
    coaching: Coaching
    window: StrokeWindow
    duration: float


def analyze(
    video_path: str,
    out_dir: str,
    *,
    estimator: PoseEstimator | None = None,
    coaching_client=None,
) -> AnalysisResult:
    estimator = estimator or MediaPipePoseEstimator()

    raw_frames = read_frames(video_path)
    pose_frames = estimator.estimate_all(raw_frames)

    window = detect_serve_window(pose_frames)
    stroke_type = classify_stroke(pose_frames, window)
    if stroke_type != StrokeType.SERVE:
        raise NotAServe(
            "This clip does not look like a serve. Film a serve from the side, "
            "full body in frame."
        )

    metrics = compute_serve_metrics(pose_frames, window)
    score = score_serve(metrics)
    overlay_paths = render_overlays(video_path, pose_frames, window, out_dir)
    coaching = generate_coaching(stroke_type, score.factors, score, client=coaching_client)
    duration = raw_frames[-1].timestamp

    return AnalysisResult(
        stroke_type=stroke_type,
        score=score,
        metrics=score.factors,
        overlay_paths=overlay_paths,
        coaching=coaching,
        window=window,
        duration=duration,
    )
