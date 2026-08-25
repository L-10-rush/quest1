"""RapidFuzz sliding-window PhraseMatcher (see approach.md §4 and §6).

STATUS: scaffold only -- `match()` is intentionally left unimplemented.
This is the core algorithmic judgment call of the whole assignment (window
sizing, scorer choice, threshold behaviour, tie-breaking, uncertainty
classification) and is exactly the kind of decision the evaluators want to
see you reason through and defend -- so it's deliberately not solved here.
The intended approach is documented step by step below.
"""

from __future__ import annotations

import logging

from src.exceptions import MatchingError
from src.matching.base import MatchCandidate, MatchResult, PhraseMatcher
from src.transcription.base import TranscriptResult

logger = logging.getLogger(__name__)


class FuzzyMatcher(PhraseMatcher):
    """Slides a window of N words across the transcript, scoring each
    window against the target phrase with RapidFuzz."""

    def match(
        self,
        transcript: TranscriptResult,
        target_text: str,
        threshold: float,
        window_size: int | None = None,
    ) -> MatchResult:
        """Find the transcript span that best matches `target_text`.

        TODO (not implemented -- fill this in):

        1. If `not transcript.words`: raise
           `MatchingError("transcript has no words to search")`.
        2. Default `window_size` to `len(target_text.split())` if None --
           consider also trying window_size +/-1 per position to absorb
           ASR insertions/deletions (e.g. a mis-heard filler word), and
           keep whichever width scores higher at each position.
        3. For each starting word index i, build the candidate string by
           joining `window_size` consecutive words' `.text`, and score it
           against `target_text` with
           `rapidfuzz.fuzz.token_sort_ratio(candidate, target_text)`
           (order-tolerant, good default for spoken phrase matching --
           swap for `fuzz.ratio` if you decide strict order matters more
           than robustness to ASR word-order slips).
        4. Record every window as a `MatchCandidate(matched_text=...,
           start_seconds=words[i].start_seconds,
           end_seconds=words[i+window_size-1].end_seconds, score=...,
           word_start_index=i, word_end_index=i+window_size)`.
        5. Sort all candidates by start time (so "first occurrence wins" is
           just "first in this list"). Filter to those with
           `score >= threshold` -> this is `candidates` in the result.
        6. If `candidates` is non-empty: `best = candidates[0]`
           (first-by-time among threshold-clearing spans, per approach.md
           §6), `is_uncertain = False`.
           Else: `best` = the single highest-scoring candidate across ALL
           windows (even below threshold), `is_uncertain = True`,
           `uncertainty_reason` = something like
           f"best match scored {best.score:.1f}, below threshold {threshold}"
           -- never return `best=None` for a non-empty transcript.
        7. Additionally flag `is_uncertain = True` (even if `score` cleared
           threshold) when the matched words' average ASR `confidence` is
           low -- a lucky high fuzzy-score match on low-confidence
           transcription is more likely a false positive than a real
           low-confidence transcript of the right words (approach.md §6).
        8. Return the `MatchResult`.
        """
        raise NotImplementedError(
            "FuzzyMatcher.match() is a scaffold -- implement steps 1-8 in "
            "the docstring above; this is the core matching logic to design "
            "and defend."
        )
