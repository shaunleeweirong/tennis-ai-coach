# CV Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python library + CLI that takes a video clip of a tennis serve and returns measured technique metrics, a deterministic 0–100 score, skeleton-overlay key frames, and grounded LLM coaching feedback — the core "CV pipeline" subsystem of the Tennis AI Coach (canonical PRD: `prd-tennis-ai-coach.md`).

**Architecture:** A linear, testable pipeline of focused modules: frame extraction → on-frame pose estimation (MediaPipe) → serve-window/contact segmentation → stroke classification → serve-metric computation → deterministic scoring → overlay rendering → LLM coaching (Claude). Each stage has one responsibility and a typed interface, so other strokes plug in later as new analyzers without touching the core. Pose detection and scoring are deterministic; the LLM only narrates the measured numbers.

**Tech Stack:** Python 3.11, OpenCV (`opencv-python`), MediaPipe (`mediapipe`, legacy `solutions.pose` — Apache-2.0), NumPy, Pydantic v2, the official Anthropic SDK (`anthropic`), pytest.

## Global Constraints

- **Python version:** 3.11 (MediaPipe legacy solutions support).
- **Package name:** `tennis_cv` (neutral internal name; product brand is unsettled per the spec).
- **Project root for this subsystem:** `cv-pipeline/` under the repo root `/Users/shaunlee/Desktop/apps/tennis-app/`.
- **Pose model:** MediaPipe legacy `mediapipe.solutions.pose` — 33 landmarks, normalized coords, no model-file download. Apache-2.0 (commercial-safe per the PRD validation log).
- **LLM:** Anthropic SDK only, model `claude-opus-4-8`, via `client.messages.parse(...)` with a Pydantic `output_format` (schema-validated). The LLM key is read from the `ANTHROPIC_API_KEY` env var; never hardcode it.
- **Determinism:** all measurement and scoring (frames → metrics → 0–100 score) must be deterministic and network-free. Only `coaching.py` touches the network.
- **MVP scope:** the **serve** is the only stroke analyzer built here. The classifier confirms a clip is a serve and rejects anything else (covers the PRD's "unusable / wrong stroke" error path). Forehand/backhand/volley analyzers are follow-on plans that reuse this pipeline.
- **Image coordinate convention:** OpenCV/MediaPipe image space — `x` right, `y` **down**. "Higher in the frame" means **smaller `y`**. Every height comparison in this plan follows that convention.
- **TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, commit. Frequent commits.
- **Serve metric target ranges are placeholders** to be calibrated against reference footage (PRD Open Question #6). They are defined as named constants in one place so calibration is a one-file change.

---

### Task 1: Project scaffolding + core types and geometry

**Files:**
- Create: `cv-pipeline/pyproject.toml`
- Create: `cv-pipeline/src/tennis_cv/__init__.py`
- Create: `cv-pipeline/src/tennis_cv/types.py`
- Create: `cv-pipeline/tests/__init__.py`
- Test: `cv-pipeline/tests/test_types.py`

**Interfaces:**
- Produces:
  - Landmark index constants: `NOSE=0, L_SHOULDER=11, R_SHOULDER=12, L_ELBOW=13, R_ELBOW=14, L_WRIST=15, R_WRIST=16, L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANKLE=27, R_ANKLE=28, NUM_LANDMARKS=33`
  - `@dataclass Landmark(x: float, y: float, z: float, visibility: float)`
  - `@dataclass RawFrame(index: int, timestamp: float, image: "np.ndarray")`
  - `@dataclass PoseFrame(index: int, timestamp: float, landmarks: list[Landmark], detected: bool)` with method `lm(self, idx: int) -> Landmark`
  - `@dataclass StrokeWindow(start_index: int, contact_index: int, end_index: int)`
  - `class StrokeType(str, Enum)` with `SERVE, FOREHAND, BACKHAND, VOLLEY, UNKNOWN`
  - `@dataclass MetricResult(name: str, value: float, unit: str, target_low: float, target_high: float, sub_score: float)`
  - `@dataclass StrokeScore(overall: int, factors: list[MetricResult])`
  - `class AnalysisError(Exception)`, `class NoPersonDetected(AnalysisError)`, `class NotAServe(AnalysisError)`
  - `def angle(a: Landmark, b: Landmark, c: Landmark) -> float` — interior angle in degrees at vertex `b`, using the 2D (x, y) vectors `b→a` and `b→c`.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_types.py`:

```python
import math
import pytest
from tennis_cv.types import (
    Landmark, PoseFrame, StrokeType, angle, L_WRIST, NUM_LANDMARKS,
    AnalysisError, NotAServe,
)


def _lm(x, y):
    return Landmark(x=x, y=y, z=0.0, visibility=1.0)


def test_angle_right_angle():
    # b at origin, a straight up, c straight right -> 90 degrees
    a, b, c = _lm(0, 1), _lm(0, 0), _lm(1, 0)
    assert angle(a, b, c) == pytest.approx(90.0, abs=1e-6)


def test_angle_straight_line():
    a, b, c = _lm(-1, 0), _lm(0, 0), _lm(1, 0)
    assert angle(a, b, c) == pytest.approx(180.0, abs=1e-6)


def test_poseframe_lm_accessor():
    lms = [_lm(i, i) for i in range(NUM_LANDMARKS)]
    f = PoseFrame(index=0, timestamp=0.0, landmarks=lms, detected=True)
    assert f.lm(L_WRIST).x == L_WRIST


def test_stroketype_serialises_to_string():
    assert StrokeType.SERVE == "serve"


def test_error_hierarchy():
    assert issubclass(NotAServe, AnalysisError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv'`

- [ ] **Step 3: Write `pyproject.toml`**

Create `cv-pipeline/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tennis-cv"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "opencv-python>=4.9",
    "mediapipe>=0.10.14",
    "numpy>=1.26",
    "pydantic>=2.6",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 4: Write the package init and types**

Create `cv-pipeline/src/tennis_cv/__init__.py`:

```python
"""Tennis AI Coach — computer-vision analysis pipeline."""
```

Create `cv-pipeline/tests/__init__.py` (empty file):

```python
```

Create `cv-pipeline/src/tennis_cv/types.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_types.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
cd cv-pipeline && git add pyproject.toml src/tennis_cv/__init__.py src/tennis_cv/types.py tests/__init__.py tests/test_types.py
git commit -m "feat(cv): scaffold tennis_cv package with core types and geometry"
```

---

### Task 2: Video frame extraction

**Files:**
- Create: `cv-pipeline/src/tennis_cv/frames.py`
- Test: `cv-pipeline/tests/test_frames.py`

**Interfaces:**
- Consumes: `RawFrame` from `tennis_cv.types`.
- Produces:
  - `def read_frames(video_path: str, max_frames: int | None = None) -> list[RawFrame]` — decodes the video with OpenCV; `timestamp` in seconds derived from FPS (defaults to 30.0 if FPS is unreported). Raises `tennis_cv.types.AnalysisError` if the video cannot be opened or has zero frames.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_frames.py`:

```python
import numpy as np
import cv2
import pytest
from tennis_cv.frames import read_frames
from tennis_cv.types import AnalysisError


def _write_video(path, n_frames=10, w=64, h=48, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_read_frames_returns_all_frames(tmp_path):
    vid = tmp_path / "clip.mp4"
    _write_video(vid, n_frames=10)
    frames = read_frames(str(vid))
    assert len(frames) == 10
    assert frames[0].index == 0
    assert frames[1].timestamp == pytest.approx(1 / 30, abs=1e-3)
    assert frames[0].image.shape == (48, 64, 3)


def test_read_frames_respects_max(tmp_path):
    vid = tmp_path / "clip.mp4"
    _write_video(vid, n_frames=10)
    frames = read_frames(str(vid), max_frames=4)
    assert len(frames) == 4


def test_read_frames_missing_file_raises():
    with pytest.raises(AnalysisError):
        read_frames("/no/such/file.mp4")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.frames'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/frames.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_frames.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/frames.py tests/test_frames.py
git commit -m "feat(cv): add OpenCV video frame extraction"
```

---

### Task 3: Pose estimation (MediaPipe adapter)

**Files:**
- Create: `cv-pipeline/src/tennis_cv/pose.py`
- Test: `cv-pipeline/tests/test_pose.py`

**Interfaces:**
- Consumes: `RawFrame`, `PoseFrame`, `Landmark`, `NUM_LANDMARKS` from `tennis_cv.types`.
- Produces:
  - `def landmarks_from_mp(result, num_landmarks: int = NUM_LANDMARKS) -> tuple[list[Landmark], bool]` — pure converter from a MediaPipe pose result object to our `Landmark` list. If `result.pose_landmarks` is falsy, returns `([zero landmarks], False)`.
  - `class PoseEstimator(Protocol)` with `def estimate_all(self, raw_frames: list[RawFrame]) -> list[PoseFrame]`.
  - `class MediaPipePoseEstimator` implementing `estimate_all` using `mediapipe.solutions.pose.Pose`.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_pose.py`:

```python
from tennis_cv.pose import landmarks_from_mp
from tennis_cv.types import NUM_LANDMARKS


class _FakeLm:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeLandmarkList:
    def __init__(self, lms):
        self.landmark = lms


class _FakeResult:
    def __init__(self, lms):
        self.pose_landmarks = _FakeLandmarkList(lms) if lms else None


def test_converts_landmarks_when_detected():
    lms = [_FakeLm(0.1 * i, 0.2 * i, 0.0, 0.9) for i in range(NUM_LANDMARKS)]
    out, detected = landmarks_from_mp(_FakeResult(lms))
    assert detected is True
    assert len(out) == NUM_LANDMARKS
    assert out[5].x == 0.5
    assert out[5].visibility == 0.9


def test_missing_pose_returns_not_detected():
    out, detected = landmarks_from_mp(_FakeResult(None))
    assert detected is False
    assert len(out) == NUM_LANDMARKS
    assert all(lm.visibility == 0.0 for lm in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_pose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.pose'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/pose.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_pose.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/pose.py tests/test_pose.py
git commit -m "feat(cv): add MediaPipe pose estimator with pure converter"
```

---

### Task 4: Serve-window segmentation

**Files:**
- Create: `cv-pipeline/src/tennis_cv/segmentation.py`
- Test: `cv-pipeline/tests/test_segmentation.py`

**Interfaces:**
- Consumes: `PoseFrame`, `StrokeWindow`, landmark constants, `NoPersonDetected` from `tennis_cv.types`.
- Produces:
  - `def dominant_wrist_index(frames: list[PoseFrame]) -> int` — returns `L_WRIST` or `R_WRIST`, whichever reaches the highest point (minimum `y`) across detected frames.
  - `def detect_serve_window(frames: list[PoseFrame]) -> StrokeWindow` — `contact_index` is the frame where the dominant wrist is highest (min `y`); `start_index` is the frame of the dominant wrist's lowest point (max `y`) at or before contact; `end_index` is the last frame. Raises `NoPersonDetected` if fewer than 3 frames are detected.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_segmentation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.segmentation'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/segmentation.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_segmentation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/segmentation.py tests/test_segmentation.py
git commit -m "feat(cv): add serve-window and contact-frame segmentation"
```

---

### Task 5: Stroke classification (serve confirmation)

**Files:**
- Create: `cv-pipeline/src/tennis_cv/classify.py`
- Test: `cv-pipeline/tests/test_classify.py`

**Interfaces:**
- Consumes: `PoseFrame`, `StrokeWindow`, `StrokeType`, landmark constants from `tennis_cv.types`; `dominant_wrist_index` from `tennis_cv.segmentation`.
- Produces:
  - `def classify_stroke(frames: list[PoseFrame], window: StrokeWindow) -> StrokeType` — returns `StrokeType.SERVE` when, at the contact frame, the dominant wrist is clearly above the nose (`wrist.y < nose.y - SERVE_WRIST_ABOVE_NOSE_MARGIN`); otherwise `StrokeType.UNKNOWN`.
  - Constant `SERVE_WRIST_ABOVE_NOSE_MARGIN = 0.05`.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_classify.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.classify'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/classify.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_classify.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/classify.py tests/test_classify.py
git commit -m "feat(cv): add serve-confirmation stroke classifier"
```

---

### Task 6: Serve metric computation

**Files:**
- Create: `cv-pipeline/src/tennis_cv/metrics.py`
- Test: `cv-pipeline/tests/test_metrics.py`

**Interfaces:**
- Consumes: `PoseFrame`, `StrokeWindow`, `MetricResult`, landmark constants, `angle` from `tennis_cv.types`; `dominant_wrist_index` from `tennis_cv.segmentation`.
- Produces:
  - Target-range constants (placeholders, calibration is Open Question #6):
    `KNEE_FLEXION_TARGET = (110.0, 140.0)`, `CONTACT_HEIGHT_TARGET = (1.5, 2.5)`, `ARM_EXTENSION_TARGET = (160.0, 180.0)`.
  - `def compute_serve_metrics(frames: list[PoseFrame], window: StrokeWindow) -> list[MetricResult]` returning three `MetricResult`s (with `sub_score=0.0`, filled later by scoring):
    - `"knee_flexion"` (deg): the **minimum** knee angle (hip–knee–ankle, dominant-leg side inferred from dominant wrist) across the window — deepest loading bend.
    - `"contact_height"` (ratio): at the contact frame, `(hip_y − wrist_y) / (hip_y − shoulder_y)` on the dominant side — how far above the shoulders the wrist reaches.
    - `"arm_extension"` (deg): elbow angle (shoulder–elbow–wrist) on the dominant side at the contact frame.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_metrics.py`:

```python
import pytest
from tennis_cv.metrics import compute_serve_metrics
from tennis_cv.types import (
    Landmark, PoseFrame, StrokeWindow, NUM_LANDMARKS,
    NOSE, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE,
)


def _serve_frame(index, *, knee_angle_pts, shoulder, elbow, wrist, hip):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    hip_pt, knee_pt, ankle_pt = knee_angle_pts
    lms[R_HIP] = Landmark(*hip_pt, 0.0, 1.0)
    lms[R_KNEE] = Landmark(*knee_pt, 0.0, 1.0)
    lms[R_ANKLE] = Landmark(*ankle_pt, 0.0, 1.0)
    lms[R_SHOULDER] = Landmark(*shoulder, 0.0, 1.0)
    lms[R_ELBOW] = Landmark(*elbow, 0.0, 1.0)
    lms[R_WRIST] = Landmark(*wrist, 0.0, 1.0)
    # Dominant-wrist detection needs the right wrist to be the higher one:
    lms[NOSE] = Landmark(0.5, 0.4, 0.0, 1.0)
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=True)


def test_computes_three_named_metrics_with_expected_values():
    # Contact frame: arm straight up (extension ~180), wrist above shoulder.
    # hip_y=0.6, shoulder_y=0.4, wrist_y=0.1 -> contact_height = (0.6-0.1)/(0.6-0.4) = 2.5
    contact = _serve_frame(
        0,
        knee_angle_pts=((0.5, 0.6), (0.5, 0.75), (0.5, 0.9)),  # straight leg ~180
        shoulder=(0.5, 0.4),
        elbow=(0.5, 0.25),
        wrist=(0.5, 0.1),
        hip=(0.5, 0.6),
    )
    # Loading frame: deep knee bend ~90 degrees (ankle directly below knee, hip forward)
    loading = _serve_frame(
        1,
        knee_angle_pts=((0.65, 0.6), (0.5, 0.7), (0.5, 0.9)),
        shoulder=(0.5, 0.4),
        elbow=(0.5, 0.5),
        wrist=(0.5, 0.6),
        hip=(0.5, 0.6),
    )
    window = StrokeWindow(start_index=1, contact_index=0, end_index=1)
    metrics = {m.name: m for m in compute_serve_metrics([contact, loading], window)}

    assert set(metrics) == {"knee_flexion", "contact_height", "arm_extension"}
    assert metrics["contact_height"].value == pytest.approx(2.5, abs=1e-6)
    assert metrics["arm_extension"].value == pytest.approx(180.0, abs=1.0)
    # Deepest knee bend across window is the ~90-degree loading frame.
    assert metrics["knee_flexion"].value == pytest.approx(90.0, abs=5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/metrics.py`:

```python
from __future__ import annotations

from .segmentation import dominant_wrist_index
from .types import (
    L_WRIST, L_SHOULDER, L_ELBOW, L_HIP, L_KNEE, L_ANKLE,
    R_SHOULDER, R_ELBOW, R_HIP, R_KNEE, R_ANKLE,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_metrics.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/metrics.py tests/test_metrics.py
git commit -m "feat(cv): add serve technique metric computation"
```

---

### Task 7: Deterministic scoring

**Files:**
- Create: `cv-pipeline/src/tennis_cv/scoring.py`
- Test: `cv-pipeline/tests/test_scoring.py`

**Interfaces:**
- Consumes: `MetricResult`, `StrokeScore` from `tennis_cv.types`.
- Produces:
  - `def sub_score(value: float, low: float, high: float) -> float` — 100.0 when `low <= value <= high`; otherwise decreases linearly to 0.0 as the value moves one range-width outside the band: `max(0.0, 100.0 * (1 - distance_outside / width))` where `width = max(high - low, 1e-9)`.
  - `def score_serve(metrics: list[MetricResult]) -> StrokeScore` — returns a new list of `MetricResult` (same fields, `sub_score` filled in) as `factors`, and `overall` = rounded mean of the sub-scores (int 0–100).

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_scoring.py`:

```python
import pytest
from tennis_cv.scoring import sub_score, score_serve
from tennis_cv.types import MetricResult


def test_sub_score_inside_range_is_100():
    assert sub_score(125, 110, 140) == 100.0


def test_sub_score_one_width_outside_is_zero():
    # width = 30; value 30 below low -> exactly one width outside -> 0
    assert sub_score(80, 110, 140) == pytest.approx(0.0, abs=1e-6)


def test_sub_score_half_width_outside_is_50():
    assert sub_score(95, 110, 140) == pytest.approx(50.0, abs=1e-6)


def test_score_serve_fills_subscores_and_averages():
    metrics = [
        MetricResult("knee_flexion", 125, "deg", 110, 140, 0.0),   # inside -> 100
        MetricResult("contact_height", 2.0, "ratio", 1.5, 2.5, 0.0),  # inside -> 100
        MetricResult("arm_extension", 170, "deg", 160, 180, 0.0),  # inside -> 100
    ]
    result = score_serve(metrics)
    assert result.overall == 100
    assert all(f.sub_score == 100.0 for f in result.factors)
    # Source metrics are not mutated.
    assert metrics[0].sub_score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.scoring'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/scoring.py`:

```python
from __future__ import annotations

from dataclasses import replace

from .types import MetricResult, StrokeScore


def sub_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 100.0
    width = max(high - low, 1e-9)
    distance = (low - value) if value < low else (value - high)
    return max(0.0, 100.0 * (1 - distance / width))


def score_serve(metrics: list[MetricResult]) -> StrokeScore:
    factors = [
        replace(m, sub_score=sub_score(m.value, m.target_low, m.target_high))
        for m in metrics
    ]
    mean = sum(f.sub_score for f in factors) / len(factors) if factors else 0.0
    return StrokeScore(overall=round(mean), factors=factors)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_scoring.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/scoring.py tests/test_scoring.py
git commit -m "feat(cv): add deterministic 0-100 serve scoring"
```

---

### Task 8: Skeleton-overlay rendering

**Files:**
- Create: `cv-pipeline/src/tennis_cv/overlay.py`
- Test: `cv-pipeline/tests/test_overlay.py`

**Interfaces:**
- Consumes: `PoseFrame`, `StrokeWindow`, landmark constants from `tennis_cv.types`.
- Produces:
  - Constant `SKELETON_EDGES: list[tuple[int, int]]` — landmark index pairs to draw (shoulders, arms, torso, legs).
  - `def render_overlays(video_path: str, pose_frames: list[PoseFrame], window: StrokeWindow, out_dir: str) -> list[str]` — draws the skeleton on the start, contact, and end frames (read from the video by index), writes them as PNGs into `out_dir` named `frame_<index>.png`, and returns the list of written paths. Frames with `detected=False` are written without skeleton lines.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_overlay.py`:

```python
import os
import numpy as np
import cv2
from tennis_cv.overlay import render_overlays
from tennis_cv.types import Landmark, PoseFrame, StrokeWindow, NUM_LANDMARKS


def _write_video(path, n_frames=8, w=64, h=48, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def _pose(index):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=True)


def test_renders_three_key_frames(tmp_path):
    vid = tmp_path / "clip.mp4"
    _write_video(vid, n_frames=8)
    poses = [_pose(i) for i in range(8)]
    window = StrokeWindow(start_index=1, contact_index=4, end_index=7)
    out_dir = tmp_path / "overlays"
    paths = render_overlays(str(vid), poses, window, str(out_dir))
    assert len(paths) == 3
    for p in paths:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.overlay'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/overlay.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_overlay.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/overlay.py tests/test_overlay.py
git commit -m "feat(cv): add skeleton-overlay key-frame rendering"
```

---

### Task 9: LLM coaching (Claude)

**Files:**
- Create: `cv-pipeline/src/tennis_cv/coaching.py`
- Test: `cv-pipeline/tests/test_coaching.py`

**Interfaces:**
- Consumes: `MetricResult`, `StrokeScore`, `StrokeType` from `tennis_cv.types`.
- Produces:
  - Pydantic models: `Issue(title: str, why: str, how: str, priority: int)`, `Drill(name: str, description: str, addresses: str)`, `Coaching(summary: str, issues: list[Issue], drills: list[Drill])`.
  - `def build_prompt(stroke_type: StrokeType, metrics: list[MetricResult], score: StrokeScore) -> str` — a deterministic prompt embedding the measured numbers, target ranges, and sub-scores; instructs the model to ground every claim in the supplied metrics and never invent measurements.
  - `def generate_coaching(stroke_type: StrokeType, metrics: list[MetricResult], score: StrokeScore, *, client=None, model: str = "claude-opus-4-8") -> Coaching` — calls `client.messages.parse(...)` with `output_format=Coaching` and returns `.parsed_output`. If `client` is `None`, constructs `anthropic.Anthropic()`.

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_coaching.py`:

```python
from types import SimpleNamespace
from tennis_cv.coaching import build_prompt, generate_coaching, Coaching, Issue, Drill
from tennis_cv.types import MetricResult, StrokeScore, StrokeType


def _score():
    factors = [
        MetricResult("knee_flexion", 95, "deg", 110, 140, 50.0),
        MetricResult("contact_height", 2.0, "ratio", 1.5, 2.5, 100.0),
        MetricResult("arm_extension", 150, "deg", 160, 180, 0.0),
    ]
    return factors, StrokeScore(overall=50, factors=factors)


def test_build_prompt_includes_metric_values_and_targets():
    factors, score = _score()
    prompt = build_prompt(StrokeType.SERVE, factors, score)
    assert "knee_flexion" in prompt
    assert "95" in prompt
    assert "110" in prompt and "140" in prompt
    assert "50" in prompt  # overall score


def test_generate_coaching_uses_injected_client_and_returns_parsed():
    factors, score = _score()
    expected = Coaching(
        summary="Drive your legs more and finish the reach.",
        issues=[Issue(title="Shallow knee bend", why="Less power.", how="Bend deeper.", priority=1)],
        drills=[Drill(name="Pause-and-drive", description="Load, pause, explode up.", addresses="knee_flexion")],
    )

    captured = {}

    def fake_parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed_output=expected)

    fake_client = SimpleNamespace(messages=SimpleNamespace(parse=fake_parse))

    result = generate_coaching(StrokeType.SERVE, factors, score, client=fake_client)

    assert result == expected
    assert captured["model"] == "claude-opus-4-8"
    assert captured["output_format"] is Coaching
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_coaching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.coaching'`

- [ ] **Step 3: Write minimal implementation**

Create `cv-pipeline/src/tennis_cv/coaching.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

from .types import MetricResult, StrokeScore, StrokeType


class Issue(BaseModel):
    title: str
    why: str
    how: str
    priority: int


class Drill(BaseModel):
    name: str
    description: str
    addresses: str


class Coaching(BaseModel):
    summary: str
    issues: list[Issue]
    drills: list[Drill]


def build_prompt(stroke_type: StrokeType, metrics: list[MetricResult], score: StrokeScore) -> str:
    lines = [
        f"You are a tennis coach analysing a {stroke_type.value}.",
        f"Overall technique score: {score.overall}/100.",
        "",
        "Measured metrics (value, unit, target range, sub-score 0-100):",
    ]
    for m in metrics:
        lines.append(
            f"- {m.name}: {m.value:.1f} {m.unit} "
            f"(target {m.target_low:.1f}-{m.target_high:.1f}, sub-score {m.sub_score:.0f})"
        )
    lines += [
        "",
        "Ground every observation in these measured numbers. Do not invent any "
        "measurement that is not listed above. Rank issues by impact (priority 1 "
        "= most important). For each issue give a specific corrective drill.",
    ]
    return "\n".join(lines)


def generate_coaching(
    stroke_type: StrokeType,
    metrics: list[MetricResult],
    score: StrokeScore,
    *,
    client=None,
    model: str = "claude-opus-4-8",
) -> Coaching:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    prompt = build_prompt(stroke_type, metrics, score)
    response = client.messages.parse(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        output_format=Coaching,
    )
    return response.parsed_output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_coaching.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/coaching.py tests/test_coaching.py
git commit -m "feat(cv): add grounded Claude coaching with structured output"
```

---

### Task 10: Pipeline orchestration + CLI

**Files:**
- Create: `cv-pipeline/src/tennis_cv/pipeline.py`
- Create: `cv-pipeline/src/tennis_cv/cli.py`
- Test: `cv-pipeline/tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass AnalysisResult(stroke_type, score, metrics, overlay_paths, coaching, window, duration)` (types: `StrokeType`, `StrokeScore`, `list[MetricResult]`, `list[str]`, `Coaching`, `StrokeWindow`, `float`).
  - `def analyze(video_path: str, out_dir: str, *, estimator: PoseEstimator | None = None, coaching_client=None) -> AnalysisResult` — runs the full pipeline: read frames → estimate poses (default `MediaPipePoseEstimator`) → segment → classify (raise `NotAServe` if not a serve) → compute metrics → score → render overlays → generate coaching. `duration` is the last frame's timestamp.
  - `def main(argv: list[str] | None = None) -> int` in `cli.py` — parses `video` positional + `--out` option, runs `analyze`, prints the result as JSON to stdout, returns exit code 0 (or 2 on `AnalysisError`, printing the message to stderr).

- [ ] **Step 1: Write the failing test**

Create `cv-pipeline/tests/test_pipeline.py`:

```python
import numpy as np
import cv2
import pytest
from types import SimpleNamespace
from tennis_cv.pipeline import analyze
from tennis_cv.coaching import Coaching, Issue, Drill
from tennis_cv.types import (
    Landmark, PoseFrame, RawFrame, NUM_LANDMARKS, StrokeType, NotAServe,
    NOSE, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE,
)


def _write_video(path, n_frames=6, w=64, h=48, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def _serve_pose(index, wrist_y):
    lms = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS)]
    lms[NOSE] = Landmark(0.5, 0.4, 0.0, 1.0)
    lms[R_SHOULDER] = Landmark(0.5, 0.45, 0.0, 1.0)
    lms[R_ELBOW] = Landmark(0.5, 0.3, 0.0, 1.0)
    lms[R_WRIST] = Landmark(0.5, wrist_y, 0.0, 1.0)
    lms[R_HIP] = Landmark(0.5, 0.6, 0.0, 1.0)
    lms[R_KNEE] = Landmark(0.5, 0.75, 0.0, 1.0)
    lms[R_ANKLE] = Landmark(0.5, 0.9, 0.0, 1.0)
    return PoseFrame(index=index, timestamp=index / 30, landmarks=lms, detected=True)


class _FakeEstimator:
    def __init__(self, poses):
        self._poses = poses

    def estimate_all(self, raw_frames: list[RawFrame]) -> list[PoseFrame]:
        return self._poses


def _fake_coaching_client():
    coaching = Coaching(
        summary="Good extension; load the legs more.",
        issues=[Issue(title="Shallow bend", why="x", how="y", priority=1)],
        drills=[Drill(name="d", description="e", addresses="knee_flexion")],
    )
    return SimpleNamespace(
        messages=SimpleNamespace(parse=lambda **kw: SimpleNamespace(parsed_output=coaching))
    )


def test_analyze_serve_end_to_end(tmp_path):
    vid = tmp_path / "serve.mp4"
    _write_video(vid, n_frames=6)
    # Wrist rises to a clear peak (min y) at frame 3 -> above the nose (0.4).
    wrist_ys = [0.7, 0.6, 0.4, 0.1, 0.3, 0.5]
    poses = [_serve_pose(i, y) for i, y in enumerate(wrist_ys)]

    result = analyze(
        str(vid),
        str(tmp_path / "out"),
        estimator=_FakeEstimator(poses),
        coaching_client=_fake_coaching_client(),
    )

    assert result.stroke_type == StrokeType.SERVE
    assert 0 <= result.score.overall <= 100
    assert len(result.metrics) == 3
    assert len(result.overlay_paths) == 3
    assert result.coaching.summary.startswith("Good extension")
    assert result.duration == pytest.approx(5 / 30, abs=1e-3)


def test_analyze_rejects_non_serve(tmp_path):
    vid = tmp_path / "not_serve.mp4"
    _write_video(vid, n_frames=6)
    # Wrist never rises above the nose -> classifier returns UNKNOWN.
    poses = [_serve_pose(i, 0.6) for i in range(6)]
    with pytest.raises(NotAServe):
        analyze(
            str(vid),
            str(tmp_path / "out"),
            estimator=_FakeEstimator(poses),
            coaching_client=_fake_coaching_client(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cv-pipeline && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tennis_cv.pipeline'`

- [ ] **Step 3: Write the pipeline**

Create `cv-pipeline/src/tennis_cv/pipeline.py`:

```python
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
```

- [ ] **Step 4: Run pipeline test to verify it passes**

Run: `cd cv-pipeline && python -m pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the CLI**

Create `cv-pipeline/src/tennis_cv/cli.py`:

```python
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .pipeline import analyze
from .types import AnalysisError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a tennis serve clip.")
    parser.add_argument("video", help="Path to the serve video clip")
    parser.add_argument("--out", default="overlays", help="Directory for overlay images")
    args = parser.parse_args(argv)

    try:
        result = analyze(args.video, args.out)
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "stroke_type": result.stroke_type.value,
        "overall_score": result.score.overall,
        "duration": result.duration,
        "metrics": [dataclasses.asdict(m) for m in result.metrics],
        "overlay_paths": result.overlay_paths,
        "coaching": result.coaching.model_dump(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the full test suite**

Run: `cd cv-pipeline && python -m pytest -v`
Expected: PASS (all tests across all modules pass)

- [ ] **Step 7: Commit**

```bash
cd cv-pipeline && git add src/tennis_cv/pipeline.py src/tennis_cv/cli.py tests/test_pipeline.py
git commit -m "feat(cv): wire end-to-end serve analysis pipeline and CLI"
```

---

## Manual smoke test (after Task 10)

This validates the real MediaPipe + Claude path, which the unit tests stub out:

1. `cd cv-pipeline && python -m pip install -e ".[dev]"`
2. `export ANTHROPIC_API_KEY=...`
3. Record (or download) a short side-on serve clip, save as `serve.mp4`.
4. `python -m tennis_cv.cli serve.mp4 --out ./overlays`
5. Confirm: JSON prints with a 0–100 score, three metrics, three overlay PNGs in `./overlays`, and a grounded coaching summary + drills. Open the PNGs and confirm the skeleton tracks the body and the contact frame is the wrist's highest point.

This is the moment to evaluate the PRD's core bet (Success Metric: ≥90% "feedback was helpful"). If the feedback is weak, iterate on `build_prompt` and the metric target ranges before building the backend or iOS subsystems.

---

## Plan self-review notes

- **Spec coverage:** This plan covers the CV-pipeline subsystem of `prd-tennis-ai-coach.md` Phase 0 — pose estimation, stroke detect/classify, technique metrics, 0–100 score, skeleton overlays, grounded LLM feedback (US-002, US-003, FR-3/4/5/6, and the serve slice of FR-1's "no player detected" via `NoPersonDetected`). Upload/job/API/history (FR-1/2/8), persistence, and iOS belong to the **backend** and **iOS** plans (separate documents, per the decomposition).
- **Deliberate scope limit:** only the **serve** analyzer is built. The classifier rejects non-serve clips rather than labelling forehand/backhand/volley. Full four-stroke classification and per-stroke analyzers are follow-on plans that reuse Tasks 1–3, 8, 9, 10 unchanged and add `metrics`/`scoring` analyzers per stroke. This is logged so the next planner doesn't assume four-stroke coverage exists.
- **Calibration debt:** serve metric target ranges in Task 6 are placeholders (PRD Open Question #6). They live in named constants for a one-file calibration pass.
- **Type consistency:** `analyze()` (Task 10) uses `estimator.estimate_all`, `detect_serve_window`, `classify_stroke`, `compute_serve_metrics`, `score_serve`, `render_overlays`, `generate_coaching` exactly as each producing task defines them; `score.factors` (filled sub-scores) is the single source of metrics passed onward.
