"""Compute word-frequency and confidence metrics for a transcript.

Fully implemented and unit-tested: pure data crunching over
`TranscriptResult`, no external dependency. Satisfies the requirement that
each per-video result.json carries word counts/frequencies and other
transcript-level metrics, not just the single matched span.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from src.transcription.base import TranscriptResult

_WORD_CLEAN_RE = re.compile(r"[^\w']+", re.UNICODE)


@dataclass(frozen=True)
class TranscriptMetrics:
    """Aggregate statistics over an entire transcript."""

    total_words: int
    unique_words: int
    word_frequencies: dict[str, int]
    average_confidence: float
    min_confidence: float
    max_confidence: float
    transcript_duration_seconds: float
    words_per_minute: float


def _normalize(word: str) -> str:
    """Lowercase and strip surrounding punctuation for frequency counting
    (so "Stagnation," and "stagnation" count as the same word), while
    keeping internal apostrophes (so "don't" stays one token)."""
    return _WORD_CLEAN_RE.sub("", word).lower()


def compute_transcript_metrics(transcript: TranscriptResult) -> TranscriptMetrics:
    """Build a `TranscriptMetrics` from a full `TranscriptResult`.

    Returns an all-zero result for an empty transcript rather than raising
    -- an empty transcript is a valid (if useless) input for a metrics
    function; the *pipeline* is responsible for treating "no words at all"
    as an error via `MatchingError`, not this pure computation.
    """
    words = transcript.words
    if not words:
        return TranscriptMetrics(
            total_words=0,
            unique_words=0,
            word_frequencies={},
            average_confidence=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
            transcript_duration_seconds=0.0,
            words_per_minute=0.0,
        )

    frequencies: dict[str, int] = {}
    for w in words:
        normalized = _normalize(w.text)
        if not normalized:
            continue
        frequencies[normalized] = frequencies.get(normalized, 0) + 1

    confidences = [w.confidence for w in words]
    duration = words[-1].end_seconds - words[0].start_seconds

    return TranscriptMetrics(
        total_words=len(words),
        unique_words=len(frequencies),
        word_frequencies=dict(
            sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        average_confidence=statistics.fmean(confidences),
        min_confidence=min(confidences),
        max_confidence=max(confidences),
        transcript_duration_seconds=duration,
        words_per_minute=(len(words) / duration * 60.0) if duration > 0 else 0.0,
    )
