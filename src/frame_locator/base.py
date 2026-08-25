"""FrameLocator interface (Dependency Inversion boundary for stage 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.ingestion.base import VideoMetadata


@dataclass(frozen=True)
class FrameResult:
    """The extracted frame plus how it was located."""

    frame_number: int
    timestamp: str  # HH:MM:SS.sss
    timestamp_seconds: float
    image: np.ndarray  # BGR, as returned by OpenCV


class FrameLocator(ABC):
    """Extracts a single frame from a video at a given timestamp."""

    @abstractmethod
    def locate(self, video: VideoMetadata, timestamp_seconds: float) -> FrameResult:
        """Return the frame at `timestamp_seconds` into `video`.

        Implementations MUST raise `exceptions.FrameExtractionError` on
        failure (e.g. seek past end of file, corrupt frame).
        """
        raise NotImplementedError
