"""ResultStore interface (Dependency Inversion boundary for stage 6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.frame_locator.base import FrameResult
from src.ingestion.base import VideoMetadata
from src.matching.base import MatchResult
from src.metrics.transcript_metrics import TranscriptMetrics
from src.transcription.base import TranscriptResult


class ResultStore(ABC):
    """Persists one pipeline run's result to disk."""

    @abstractmethod
    def save(
        self,
        video: VideoMetadata,
        target_text: str,
        match: MatchResult,
        frame: FrameResult,
        metrics: TranscriptMetrics,
        transcript: TranscriptResult,
    ) -> Path:
        """Persist the run and return the path to the written `result.json`.

        Implementations MUST raise `exceptions.ResultPersistenceError` on
        failure (e.g. unwritable output directory).
        """
        raise NotImplementedError
