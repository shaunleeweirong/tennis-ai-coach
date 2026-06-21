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
