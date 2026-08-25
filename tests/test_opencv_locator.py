"""Verifies OpenCvFrameLocator against a synthetically generated video --
no network, no real media file needed. Each frame is rendered as a solid
color encoding its own index, so landing on the wrong frame is detectable
by pixel value, not just "did it not crash".
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.frame_locator.opencv_locator import OpenCvFrameLocator
from src.ingestion.base import VideoMetadata

FPS = 10.0
FRAME_COUNT = 50
SIZE = (64, 64)  # width, height


def _make_test_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, SIZE)
    for i in range(FRAME_COUNT):
        # Encode the frame index in the blue channel so we can assert on it.
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


class TestOpenCvFrameLocator:
    def test_locates_correct_frame_number(self, test_video: VideoMetadata):
        locator = OpenCvFrameLocator()
        # 1.0s * 10fps == frame 10
        result = locator.locate(test_video, timestamp_seconds=1.0)
        assert result.frame_number == 10
        assert result.timestamp == "00:00:01.000"

    def test_extracted_frame_content_matches_encoded_index(self, test_video: VideoMetadata):
        locator = OpenCvFrameLocator()
        result = locator.locate(test_video, timestamp_seconds=2.0)  # frame 20
        blue_channel_value = int(result.image[0, 0, 0])
        # mp4v is a lossy codec -- adjacent near-solid-color frames bleed
        # into each other slightly, so pixel values aren't byte-exact. A
        # tolerance still proves we landed on frame ~20, not e.g. frame 0
        # or frame 40 (each 10+ apart), which is what this test guards.
        assert abs(blue_channel_value - 20) <= 5

    def test_first_frame(self, test_video: VideoMetadata):
        locator = OpenCvFrameLocator()
        result = locator.locate(test_video, timestamp_seconds=0.0)
        assert result.frame_number == 0
        assert int(result.image[0, 0, 0]) == 0

    def test_missing_file_raises_frame_extraction_error(self, tmp_path: Path):
        from src.exceptions import FrameExtractionError

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
        with pytest.raises(FrameExtractionError):
            OpenCvFrameLocator().locate(missing, timestamp_seconds=0.0)
