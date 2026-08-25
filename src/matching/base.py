"""PhraseMatcher interface (Dependency Inversion boundary for stage 4).

This is where "ambiguous or uncertain" (a requirement called out explicitly
in the problem statement's Evaluation section) is decided. The contract is
deliberately strict about never returning nothing:

    A PhraseMatcher implementation MUST always return a `best` candidate
    when `transcript.words` is non-empty, even if nothing clears
    `threshold` -- in that case it sets `is_uncertain=True` and fills in
    `uncertainty_reason`. Silent failure (returning None / raising because
    "nothing matched well") is not acceptable: the pipeline always has an
    answer to report, just possibly a flagged, low-confidence one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.transcription.base import TranscriptResult


@dataclass(frozen=True)
class MatchCandidate:
    """One candidate span of the transcript scored against the target text."""

    matched_text: str
    start_seconds: float
    end_seconds: float
    score: float  # 0-100, RapidFuzz-style similarity
    word_start_index: int
    word_end_index: int  # exclusive


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching the target phrase against a transcript.

    `best` is never None for a non-empty transcript -- see module docstring.
    `candidates` holds every span at or above `threshold` (for the "multiple
    similarly-scored spans" case in approach.md §6), sorted by first
    occurrence, so the same repeated line always resolves to its first
    appearance by default.
    """

    best: MatchCandidate | None
    candidates: tuple[MatchCandidate, ...]
    is_uncertain: bool
    uncertainty_reason: str | None


class PhraseMatcher(ABC):
    """Finds the best-matching transcript span for a target phrase."""

    @abstractmethod
    def match(
        self,
        transcript: TranscriptResult,
        target_text: str,
        threshold: float,
        window_size: int | None = None,
    ) -> MatchResult:
        """Search `transcript` for `target_text`.

        Implementations MUST raise `exceptions.MatchingError` only for
        genuine failures (e.g. empty transcript) -- a low-scoring match is
        NOT a failure, it's a normal `MatchResult` with `is_uncertain=True`.
        """
        raise NotImplementedError
