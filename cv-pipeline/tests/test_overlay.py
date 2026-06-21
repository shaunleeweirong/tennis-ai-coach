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
