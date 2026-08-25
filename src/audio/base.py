"""AudioExtractor interface (Dependency Inversion boundary for stage 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.base import VideoMetadata


@dataclass(frozen=True)
class AudioAsset:
    """A mono, ASR-ready audio track extracted from the source video."""

    file_path: Path
    sample_rate_hz: int
    channels: int
    duration_seconds: float


class AudioExtractor(ABC):
    """Extracts an audio track suitable for a transcription engine."""

    @abstractmethod
    def extract(self, video: VideoMetadata, dest_dir: Path) -> AudioAsset:
        """Extract audio from `video.file_path` into `dest_dir`.

        Implementations MUST raise `exceptions.AudioExtractionError` on
        failure, never a raw subprocess/library exception.
        """
        raise NotImplementedError
