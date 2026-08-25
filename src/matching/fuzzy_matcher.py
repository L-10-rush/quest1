"""RapidFuzz sliding-window PhraseMatcher (see approach.md §4 and §6).

Slides a window of N words across the transcript, scores each window
against the target phrase with RapidFuzz's `token_sort_ratio` (order-
tolerant -- absorbs ASR word-order slips better than a strict `ratio`),
and returns the earliest span that clears the confidence threshold. If
nothing clears it, the single best-scoring span is still returned, flagged
`is_uncertain` -- see the "never return nothing" contract in matching/base.py.
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from src.exceptions import MatchingError
from src.matching.base import MatchCandidate, MatchResult, PhraseMatcher
from src.transcription.base import TranscriptResult, Word

logger = logging.getLogger(__name__)

# A matched span whose words' average ASR confidence falls below this is
# flagged uncertain even if its fuzzy score cleared `threshold` -- a high
# fuzzy score built on low-confidence transcription is more likely a lucky
# false positive than a genuine (if quietly-spoken) match. See approach.md §6.
LOW_CONFIDENCE_THRESHOLD = 0.5


class FuzzyMatcher(PhraseMatcher):
    """Sliding-window fuzzy phrase matcher over a word-level transcript."""

    def match(
        self,
        transcript: TranscriptResult,
        target_text: str,
        threshold: float,
        window_size: int | None = None,
    ) -> MatchResult:
        words = transcript.words
        if not words:
            raise MatchingError("transcript has no words to search")
        if not target_text.strip():
            raise MatchingError("target_text must not be empty")

        base_width = window_size or len(target_text.split())
        if base_width < 1:
            raise MatchingError(f"window_size must be >= 1, got {base_width}")

        candidates = self._score_all_windows(words, target_text, base_width)
        if not candidates:
            # Only possible if base_width exceeds the transcript's total
            # word count (a very short transcript vs. a long target phrase).
            raise MatchingError(
                f"target phrase ({base_width} words) is longer than the "
                f"transcript ({len(words)} words) -- no window fits"
            )

        cleared = self._select_non_overlapping(
            [c for c in candidates if c.score >= threshold]
        )

        if cleared:
            if len(cleared) > 1:
                logger.info(
                    "%d distinct spans cleared threshold %.1f -- reporting "
                    "the first occurrence, see `candidates` for the rest",
                    len(cleared),
                    threshold,
                )
            best = cleared[0]  # first-by-time among threshold-clearing spans
            avg_confidence = self._average_confidence(words, best)
            if avg_confidence < LOW_CONFIDENCE_THRESHOLD:
                return MatchResult(
                    best=best,
                    candidates=tuple(cleared),
                    is_uncertain=True,
                    uncertainty_reason=(
                        f"matched span has low average transcription "
                        f"confidence ({avg_confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD})"
                    ),
                )
            return MatchResult(
                best=best, candidates=tuple(cleared), is_uncertain=False, uncertainty_reason=None
            )

        # Nothing cleared the bar -- still return a best-effort answer.
        best = max(candidates, key=lambda c: c.score)
        return MatchResult(
            best=best,
            candidates=(),
            is_uncertain=True,
            uncertainty_reason=(
                f"best match scored {best.score:.1f}, below threshold {threshold:.1f}"
            ),
        )

    def _score_all_windows(
        self, words: tuple[Word, ...], target_text: str, base_width: int
    ) -> list[MatchCandidate]:
        """One candidate per start index: the best-scoring width among
        `base_width - 1, base_width, base_width + 1` (clamped to a valid
        range) at that position, absorbing a single ASR word insertion or
        deletion without flooding the result with near-duplicate widths."""
        widths = sorted(
            {w for w in (base_width - 1, base_width, base_width + 1) if 1 <= w <= len(words)}
        )
        target_lower = target_text.lower()

        results: list[MatchCandidate] = []
        for start in range(len(words)):
            best_at_start: MatchCandidate | None = None
            for width in widths:
                end = start + width
                if end > len(words):
                    continue
                span = words[start:end]
                candidate_text = " ".join(w.text for w in span)
                score = fuzz.token_sort_ratio(candidate_text.lower(), target_lower)
                if best_at_start is None or score > best_at_start.score:
                    best_at_start = MatchCandidate(
                        matched_text=candidate_text,
                        start_seconds=span[0].start_seconds,
                        end_seconds=span[-1].end_seconds,
                        score=score,
                        word_start_index=start,
                        word_end_index=end,
                    )
            if best_at_start is not None:
                results.append(best_at_start)
        return results

    @staticmethod
    def _overlaps(a: MatchCandidate, b: MatchCandidate) -> bool:
        return a.word_start_index < b.word_end_index and b.word_start_index < a.word_end_index

    def _select_non_overlapping(
        self, candidates: list[MatchCandidate]
    ) -> list[MatchCandidate]:
        """Collapse a stride-1 sliding window's near-duplicate overlapping
        hits (the same phrase shifted by one word) into one candidate per
        distinct occurrence, keeping the highest-scoring span for each.

        Without this, a single real match produces several candidates
        (start-1, start, start+1, ...) that would misleadingly read as
        "multiple similarly-scored spans" (the ambiguity case in
        approach.md §6) when it's really just one match.
        """
        by_score_desc = sorted(candidates, key=lambda c: c.score, reverse=True)
        kept: list[MatchCandidate] = []
        for candidate in by_score_desc:
            if not any(self._overlaps(candidate, k) for k in kept):
                kept.append(candidate)
        return sorted(kept, key=lambda c: c.word_start_index)

    @staticmethod
    def _average_confidence(words: tuple[Word, ...], candidate: MatchCandidate) -> float:
        span = words[candidate.word_start_index : candidate.word_end_index]
        if not span:
            return 0.0
        return sum(w.confidence for w in span) / len(span)
