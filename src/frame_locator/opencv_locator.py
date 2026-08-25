"""OpenCV-backed FrameLocator.

Fully implemented (unlike the ingestion/audio/transcription/matching
stages) because it needs no network access, no ML model, and no
platform-specific external binary -- it's pure, verifiable OpenCV file
I/O, and is covered by `tests/test_opencv_locator.py` against a small
synthetically-generated video.
"""

from __future__ import annotations

import logging

import cv2

from src.exceptions import FrameExtractionError
from src.frame_locator.base import FrameLocator, FrameResult
from src.ingestion.base import VideoMetadata
from src.utils.timestamp import format_timestamp, seconds_to_frame_number

logger = logging.getLogger(__name__)


class OpenCvFrameLocator(FrameLocator):
    """Seeks a video with OpenCV and reads the target frame.

    Direct `CAP_PROP_POS_FRAMES` seeking can land a few frames off on some
    codecs that only seek to keyframes (a well-known OpenCV/ffmpeg
    limitation) -- to guard against that we verify the position we landed
    on and, if it's off, fall back to sequential reads from the nearest
    reliable point rather than silently returning the wrong frame.
    """

    #: If the seeked-to position drifts further than this from the target,
    #: fall back to slower-but-exact sequential reading.
    _MAX_SEEK_DRIFT_FRAMES = 2

    def locate(self, video: VideoMetadata, timestamp_seconds: float) -> FrameResult:
        target_frame = seconds_to_frame_number(timestamp_seconds, video.fps)

        cap = cv2.VideoCapture(str(video.file_path))
        if not cap.isOpened():
            raise FrameExtractionError(f"could not open video file: {video.file_path}")

        try:
            image = self._read_frame(cap, target_frame, video)
        finally:
            cap.release()

        return FrameResult(
            frame_number=target_frame,
            timestamp=format_timestamp(timestamp_seconds),
            timestamp_seconds=timestamp_seconds,
            image=image,
        )

    def _read_frame(self, cap: cv2.VideoCapture, target_frame: int, video: VideoMetadata):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
        landed_at = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if abs(landed_at - target_frame) > self._MAX_SEEK_DRIFT_FRAMES:
            logger.debug(
                "direct seek drifted (%d requested, %d landed) -- falling "
                "back to sequential read from a safe checkpoint",
                target_frame,
                landed_at,
            )
            checkpoint = max(0, target_frame - self._MAX_SEEK_DRIFT_FRAMES * 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(checkpoint))
            frame = None
            for _ in range(target_frame - checkpoint + 1):
                ok, frame = cap.read()
                if not ok:
                    raise FrameExtractionError(
                        f"failed to sequentially read up to frame {target_frame} "
                        f"in {video.file_path}"
                    )
            return frame

        ok, frame = cap.read()
        if not ok:
            raise FrameExtractionError(
                f"failed to read frame {target_frame} from {video.file_path} "
                f"(video has ~{video.duration_seconds * video.fps:.0f} frames)"
            )
        return frame
