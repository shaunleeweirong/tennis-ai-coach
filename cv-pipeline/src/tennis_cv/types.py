from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# MediaPipe Pose landmark indices (33-point model).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NUM_LANDMARKS = 33


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class RawFrame:
    index: int
    timestamp: float
    image: "np.ndarray"


@dataclass
class PoseFrame:
    index: int
    timestamp: float
    landmarks: list[Landmark]
    detected: bool

    def lm(self, idx: int) -> Landmark:
        return self.landmarks[idx]


@dataclass
class StrokeWindow:
    start_index: int
    contact_index: int
    end_index: int


class StrokeType(str, Enum):
    SERVE = "serve"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    VOLLEY = "volley"
    UNKNOWN = "unknown"


@dataclass
class MetricResult:
    name: str
    value: float
    unit: str
    target_low: float
    target_high: float
    sub_score: float


@dataclass
class StrokeScore:
    overall: int
    factors: list[MetricResult]


class AnalysisError(Exception):
    """Base class for recoverable, user-facing analysis failures."""


class NoPersonDetected(AnalysisError):
    pass


class NotAServe(AnalysisError):
    pass


def angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Interior angle in degrees at vertex ``b`` using 2D (x, y) vectors."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag == 0.0:
        return 0.0
    cosv = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cosv))
