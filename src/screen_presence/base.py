"""ScreenPresenceDetector interface (Dependency Inversion boundary for stage 6).

Answers the more literal reading of the problem statement's "on-screen
dialogue": not just where a line was SAID (stage 4's job), but whether the
speaking character was visibly on camera saying it -- as opposed to
voice-over, off-camera narration, or a still shot of someone who isn't the
one talking. See approach.md for why this is a *heuristic*, not a trained
active-speaker-detection model, and the tradeoff that decision makes
against a hosted/gated-model approach.

The contract mirrors PhraseMatcher's (matching/base.py): a genuinely
ambiguous read is a normal result with status="uncertain" and a reason,
never an exception. `ScreenPresenceError` is reserved for cases where
verification could not run at all (e.g. the video file can't be opened).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from src.ingestion.base import VideoMetadata

ScreenPresenceStatus = Literal["on_screen", "off_screen", "uncertain"]


@dataclass(frozen=True)
class ScreenPresenceResult:
    """Best-effort verdict on whether a speaking face was visible during a
    matched dialogue's time window."""

    status: ScreenPresenceStatus
    confidence: float  # 0-1
    reason: str
    face_ratio: float  # fraction of readable sampled frames with a detected face, 0-1
    mouth_motion_score: float  # 0-1, normalized frame-to-frame mouth-region motion
    frames_sampled: int  # frames actually read, not counting failed seeks/reads


class ScreenPresenceDetector(ABC):
    """Verifies whether a speaking face was visible on screen during a
    matched dialogue's time window."""

    @abstractmethod
    def verify(
        self, video: VideoMetadata, start_seconds: float, end_seconds: float
    ) -> ScreenPresenceResult:
        """Sample frames across `[start_seconds, end_seconds]` and decide
        whether a speaking face was visibly on camera.

        Implementations MUST always return a `ScreenPresenceResult` --
        including for an inconclusive read (`status="uncertain"`) -- and
        MUST raise `exceptions.ScreenPresenceError` only for a genuine
        failure to even attempt verification (e.g. the video file can't be
        opened), never for "couldn't find a face."
        """
        raise NotImplementedError
