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

# `token_sort_ratio` is sensitive to string length in ways that don't
# always track match quality: a truncated window can occasionally score a
# couple of points higher than the full-width window even when the full
# width is the genuinely correct span (e.g. dropping a matched word can
# score higher than keeping it alongside one substitution). Only switch
# away from `base_width` when a wider/narrower window wins by more than
# this margin -- a small margin absorbs that scoring noise while a large
# jump (an actual inserted/dropped word, see approach.md §4) still wins.
WIDTH_PREFERENCE_MARGIN = 3.0


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
            [c for c in candidates if c.score >= threshold], base_width
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
        """One candidate per start index, picked among widths
        `base_width - 1, base_width, base_width + 1` (clamped to a valid
        range) via `_pick_width` -- absorbing a single ASR word insertion
        or deletion without flooding the result with near-duplicate widths,
        while resisting `token_sort_ratio`'s length-sensitivity noise."""
        widths = sorted(
            {w for w in (base_width - 1, base_width, base_width + 1) if 1 <= w <= len(words)}
        )
        target_lower = target_text.lower()

        results: list[MatchCandidate] = []
        for start in range(len(words)):
            scored_by_width: dict[int, tuple[float, tuple[Word, ...], str]] = {}
            for width in widths:
                end = start + width
                if end > len(words):
                    continue
                span = words[start:end]
                candidate_text = " ".join(w.text for w in span)
                score = fuzz.token_sort_ratio(candidate_text.lower(), target_lower)
                scored_by_width[width] = (score, span, candidate_text)

            if not scored_by_width:
                continue

            width = self._pick_width(scored_by_width, base_width)
            score, span, candidate_text = scored_by_width[width]
            results.append(
                MatchCandidate(
                    matched_text=candidate_text,
                    start_seconds=span[0].start_seconds,
                    end_seconds=span[-1].end_seconds,
                    score=score,
                    word_start_index=start,
                    word_end_index=start + width,
                )
            )
        return results

    @staticmethod
    def _pick_width(
        scored_by_width: dict[int, tuple[float, tuple[Word, ...], str]], base_width: int
    ) -> int:
        """Pick `base_width` unless a different width clears it by more
        than `WIDTH_PREFERENCE_MARGIN` -- see the constant's docstring."""
        best_width = max(scored_by_width, key=lambda w: scored_by_width[w][0])
        if best_width == base_width or base_width not in scored_by_width:
            return best_width

        best_score = scored_by_width[best_width][0]
        base_score = scored_by_width[base_width][0]
        if best_score > base_score + WIDTH_PREFERENCE_MARGIN:
            return best_width
        return base_width

    @staticmethod
    def _overlaps(a: MatchCandidate, b: MatchCandidate) -> bool:
        return a.word_start_index < b.word_end_index and b.word_start_index < a.word_end_index

    def _select_non_overlapping(
        self, candidates: list[MatchCandidate], base_width: int
    ) -> list[MatchCandidate]:
        """Collapse a stride-1 sliding window's near-duplicate overlapping
        hits (the same phrase shifted by one word, possibly at a different
        width -- see `_pick_width`) into one candidate per distinct
        occurrence.

        Without this, a single real match produces several candidates
        (start-1, start, start+1, ...) that would misleadingly read as
        "multiple similarly-scored spans" (the ambiguity case in
        approach.md §6) when it's really just one match.

        Ranks by score after subtracting a small penalty for how far a
        candidate's width strays from `base_width` -- the same
        length-sensitivity noise `_pick_width` guards against at a single
        start index (see its docstring) can otherwise make an overlapping
        candidate at the *wrong* start index win purely from a couple of
        points of scoring noise, e.g. truncating a leading word entirely
        rather than keeping it alongside one substitution. The penalty is
        half the width-preference margin per word of deviation: small
        enough that a genuine insertion/deletion (which swings the raw
        score by ~20 points, see `WIDTH_PREFERENCE_MARGIN`'s docstring)
        still wins, large enough to break few-point ties in `base_width`'s
        favor.
        """

        def priority(c: MatchCandidate) -> tuple[float, float]:
            width = c.word_end_index - c.word_start_index
            adjusted_score = c.score - (WIDTH_PREFERENCE_MARGIN / 2) * abs(width - base_width)
            return (adjusted_score, c.score)

        by_priority_desc = sorted(candidates, key=priority, reverse=True)
        kept: list[MatchCandidate] = []
        for candidate in by_priority_desc:
            if not any(self._overlaps(candidate, k) for k in kept):
                kept.append(candidate)
        return sorted(kept, key=lambda c: c.word_start_index)

    @staticmethod
    def _average_confidence(words: tuple[Word, ...], candidate: MatchCandidate) -> float:
        span = words[candidate.word_start_index : candidate.word_end_index]
        if not span:
            return 0.0
        return sum(w.confidence for w in span) / len(span)
