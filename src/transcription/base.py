"""TranscriptionEngine interface (Dependency Inversion boundary for stage 3).

Both `WhisperXEngine` (default) and `VoskEngine` (lightweight fallback)
implement this same contract -- `pipeline.py` never knows or cares which
one produced the transcript (Liskov Substitution).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.audio.base import AudioAsset


@dataclass(frozen=True)
class Word:
    """A single transcribed word with its position in time.

    `confidence` is in [0, 1]; engines that don't natively expose a
    per-word confidence should report their best estimate (e.g. the
    segment-level score) rather than a fabricated 1.0 -- downstream
    uncertainty handling depends on this being meaningful.
    """

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True)
class DialogueSegment:
    """One continuous spoken utterance -- what a human would call "a line
    of dialogue", as opposed to a single `Word`.

    Engines derive this from their own natural utterance boundaries
    (WhisperX: Whisper's own segment decoding, one span per pause/break;
    Vosk: one per `AcceptWaveform`-completed endpoint) rather than any
    re-segmentation logic in this codebase -- it's surfaced, not invented.
    """

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float  # average of this segment's word confidences


@dataclass(frozen=True)
class TranscriptResult:
    """Full transcript of one audio asset.

    `words` is the word-level, timestamp-aligned view the matcher
    searches (see matching/fuzzy_matcher.py). `segments` is the coarser
    "every line spoken in the video" view -- what
    `output/json_store.py` persists as the full transcript, independent
    of whatever single phrase was searched for. Defaults to `()` so
    existing test fixtures that only care about `words` don't need to
    change.
    """

    words: tuple[Word, ...]
    language: str
    engine_name: str
    model_name: str
    segments: tuple[DialogueSegment, ...] = ()


class TranscriptionEngine(ABC):
    """Transcribes audio to a word-level, timestamp-aligned transcript."""

    @abstractmethod
    def transcribe(self, audio: AudioAsset, language: str) -> TranscriptResult:
        """Transcribe `audio`, returning words in chronological order.

        Implementations MUST raise `exceptions.TranscriptionError` on
        failure, never a raw library exception.
        """
        raise NotImplementedError
