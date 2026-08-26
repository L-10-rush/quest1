"""Unit tests for OpenCvScreenPresenceDetector.

Split like the other CV-boundary tests in this project: `_decide()` and
`_motion_score()` are pure functions tested directly with plain numbers/
arrays (no OpenCV involved), and `verify()` is tested against a small real
synthesized video (real seeks, real reads) with `_detect_mouth_crop`
monkeypatched -- OpenCV's own face-detection accuracy isn't this project's
code to test; our sampling/aggregation/decision logic is.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.exceptions import ScreenPresenceError
from src.ingestion.base import VideoMetadata
from src.screen_presence.opencv_detector import OpenCvScreenPresenceDetector

FPS = 10.0
FRAME_COUNT = 50
SIZE = (64, 64)  # width, height


def _make_test_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, SIZE)
    for i in range(FRAME_COUNT):
        frame = np.full((SIZE[1], SIZE[0], 3), (i % 256, 0, 0), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def test_video(tmp_path: Path) -> VideoMetadata:
    video_path = tmp_path / "synthetic.mp4"
    _make_test_video(video_path)
    return VideoMetadata(
        video_id="test",
        source_url="https://example.com/test",
        file_path=video_path,
        title="synthetic test video",
        duration_seconds=FRAME_COUNT / FPS,
        fps=FPS,
        width=SIZE[0],
        height=SIZE[1],
    )


class TestDecide:
    """`_decide()` is the pure status/confidence/reason policy -- no cv2
    involved, so every threshold band is exercised directly."""

    def test_low_face_ratio_is_off_screen(self):
        status, confidence, reason = OpenCvScreenPresenceDetector._decide(
            face_ratio=0.0, mouth_motion_score=0.0
        )
        assert status == "off_screen"
        assert confidence == 1.0
        assert "no face" in reason

    def test_mid_face_ratio_is_uncertain(self):
        status, _confidence, reason = OpenCvScreenPresenceDetector._decide(
            face_ratio=0.3, mouth_motion_score=0.5
        )
        assert status == "uncertain"
        assert "only" in reason

    def test_high_face_ratio_but_static_mouth_is_uncertain(self):
        status, _confidence, reason = OpenCvScreenPresenceDetector._decide(
            face_ratio=1.0, mouth_motion_score=0.0
        )
        assert status == "uncertain"
        assert "static shot" in reason or "silent listener" in reason

    def test_high_face_ratio_with_motion_is_on_screen(self):
        status, confidence, reason = OpenCvScreenPresenceDetector._decide(
            face_ratio=1.0, mouth_motion_score=0.08
        )
        assert status == "on_screen"
        assert confidence > 0.5
        assert "mouth movement" in reason

    def test_confidence_is_bounded_at_one(self):
        _status, confidence, _reason = OpenCvScreenPresenceDetector._decide(
            face_ratio=1.0, mouth_motion_score=10.0  # absurdly high on purpose
        )
        assert confidence == 1.0

    def test_boundary_exactly_at_min_face_ratio_on_screen_counts_as_on_screen_track(self):
        # At the boundary itself, ratio is no longer "< threshold" so it
        # takes the on_screen branch (given sufficient motion).
        status, _confidence, _reason = OpenCvScreenPresenceDetector._decide(
            face_ratio=OpenCvScreenPresenceDetector._MIN_FACE_RATIO_ON_SCREEN,
            mouth_motion_score=OpenCvScreenPresenceDetector._MOUTH_MOTION_SATURATION,
        )
        assert status == "on_screen"


class TestMotionScore:
    def test_no_crops_is_zero(self):
        assert OpenCvScreenPresenceDetector._motion_score([]) == 0.0

    def test_single_crop_is_zero(self):
        crop = np.zeros((32, 32), dtype=np.uint8)
        assert OpenCvScreenPresenceDetector._motion_score([crop]) == 0.0

    def test_identical_crops_have_zero_motion(self):
        crop = np.full((32, 32), 128, dtype=np.uint8)
        assert OpenCvScreenPresenceDetector._motion_score([crop, crop, crop]) == 0.0

    def test_maximally_different_crops_score_near_one(self):
        black = np.zeros((32, 32), dtype=np.uint8)
        white = np.full((32, 32), 255, dtype=np.uint8)
        score = OpenCvScreenPresenceDetector._motion_score([black, white, black, white])
        assert score == pytest.approx(1.0)


class TestVerify:
    def test_face_consistently_detected_with_motion_yields_on_screen(
        self, monkeypatch, test_video: VideoMetadata
    ):
        black = np.zeros((32, 32), dtype=np.uint8)
        white = np.full((32, 32), 255, dtype=np.uint8)
        crops = [black, white] * 4  # alternating -- clear motion signal
        calls = iter(crops)
        monkeypatch.setattr(
            OpenCvScreenPresenceDetector, "_detect_mouth_crop", lambda self, frame: next(calls)
        )

        result = OpenCvScreenPresenceDetector().verify(test_video, start_seconds=1.0, end_seconds=2.0)

        assert result.status == "on_screen"
        assert result.face_ratio == 1.0
        assert result.frames_sampled == OpenCvScreenPresenceDetector._SAMPLE_COUNT

    def test_no_face_ever_detected_yields_off_screen(self, monkeypatch, test_video: VideoMetadata):
        monkeypatch.setattr(
            OpenCvScreenPresenceDetector, "_detect_mouth_crop", lambda self, frame: None
        )

        result = OpenCvScreenPresenceDetector().verify(test_video, start_seconds=1.0, end_seconds=2.0)

        assert result.status == "off_screen"
        assert result.face_ratio == 0.0

    def test_face_detected_but_static_yields_uncertain(self, monkeypatch, test_video: VideoMetadata):
        same_crop = np.full((32, 32), 100, dtype=np.uint8)
        monkeypatch.setattr(
            OpenCvScreenPresenceDetector, "_detect_mouth_crop", lambda self, frame: same_crop
        )

        result = OpenCvScreenPresenceDetector().verify(test_video, start_seconds=1.0, end_seconds=2.0)

        assert result.status == "uncertain"
        assert result.face_ratio == 1.0
        assert result.mouth_motion_score == 0.0

    def test_missing_file_raises_screen_presence_error(self, tmp_path: Path):
        missing = VideoMetadata(
            video_id="missing",
            source_url="https://example.com/missing",
            file_path=tmp_path / "does_not_exist.mp4",
            title="missing",
            duration_seconds=1.0,
            fps=FPS,
            width=SIZE[0],
            height=SIZE[1],
        )
        with pytest.raises(ScreenPresenceError):
            OpenCvScreenPresenceDetector().verify(missing, start_seconds=0.0, end_seconds=1.0)

    def test_window_is_padded_and_clamped_to_video_duration(
        self, monkeypatch, test_video: VideoMetadata
    ):
        """A match right at the very end of the video shouldn't try to seek
        past it just because of the trailing pad."""
        monkeypatch.setattr(
            OpenCvScreenPresenceDetector, "_detect_mouth_crop", lambda self, frame: None
        )

        result = OpenCvScreenPresenceDetector().verify(
            test_video,
            start_seconds=test_video.duration_seconds - 0.1,
            end_seconds=test_video.duration_seconds,
        )

        # still gets *some* readable frames rather than erroring out
        assert result.frames_sampled > 0
