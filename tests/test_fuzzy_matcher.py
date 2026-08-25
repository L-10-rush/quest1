"""Unit tests for FuzzyMatcher: the sliding-window RapidFuzz matcher.

Covers the "how do you determine where to look" / "how do you handle
ambiguity or uncertainty" requirements from the problem statement:
exact/fuzzy matches, ASR word insertion/deletion tolerance, low-confidence
flagging, repeated-phrase disambiguation, and the never-return-nothing
contract for a transcript with no good match.
"""

import pytest

from src.exceptions import MatchingError
from src.matching.fuzzy_matcher import FuzzyMatcher
from src.transcription.base import TranscriptResult, Word

WORD_DURATION = 0.3


def _words(texts: list[str], start_at: float = 0.0, confidence: float = 0.9) -> list[Word]:
    """Build a list of sequential, back-to-back Words (no gaps) from plain
    text tokens -- keeps test fixtures readable as a sentence string."""
    words = []
    t = start_at
    for text in texts:
        words.append(
            Word(text=text, start_seconds=t, end_seconds=t + WORD_DURATION, confidence=confidence)
        )
        t += WORD_DURATION
    return words


def _transcript(words: list[Word]) -> TranscriptResult:
    return TranscriptResult(
        words=tuple(words), language="en", engine_name="test", model_name="test"
    )


TARGET = "My mind rebels at stagnation"


class TestExactAndFuzzyMatch:
    def test_exact_match_locates_correct_span(self):
        words = _words(TARGET.split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert not result.is_uncertain
        assert result.best.score == 100.0
        assert result.best.start_seconds == words[0].start_seconds
        assert result.best.end_seconds == words[-1].end_seconds
        assert result.best.word_start_index == 0
        assert result.best.word_end_index == 5

    def test_minor_asr_substitution_still_clears_default_threshold(self):
        words = _words("My mynd rebels at stagnation".split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert not result.is_uncertain
        assert result.best.matched_text == "My mynd rebels at stagnation"

    def test_match_is_case_insensitive(self):
        words = _words("MY MIND REBELS AT STAGNATION".split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert result.best.score == 100.0


class TestWindowWidthTolerance:
    def test_inserted_filler_word_is_absorbed(self):
        # ASR heard an extra "uh" the speaker didn't script -- base_width=5
        # (len(TARGET.split())) but the real span is 6 words wide.
        words = _words("My mind uh rebels at stagnation".split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert not result.is_uncertain
        assert result.best.word_end_index - result.best.word_start_index == 6

    def test_dropped_word_is_absorbed(self):
        # ASR missed "rebels" entirely -- real span is 4 words wide.
        words = _words("My mind at stagnation".split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert not result.is_uncertain
        assert result.best.word_end_index - result.best.word_start_index == 4


class TestUncertaintyHandling:
    def test_no_good_match_returns_best_effort_flagged_uncertain(self):
        words = _words("the weather today is quite nice actually".split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert result.best is not None  # never None, even with no good match
        assert result.is_uncertain
        assert "below threshold" in result.uncertainty_reason
        assert result.candidates == ()  # nothing cleared the bar

    def test_low_confidence_match_flagged_uncertain_despite_high_score(self):
        words = _words(TARGET.split(), confidence=0.2)
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert result.best.score == 100.0  # the text match itself is fine
        assert result.is_uncertain
        assert "confidence" in result.uncertainty_reason

    def test_high_confidence_match_is_not_flagged_uncertain(self):
        words = _words(TARGET.split(), confidence=0.95)
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert not result.is_uncertain
        assert result.uncertainty_reason is None


class TestRepeatedPhrase:
    def test_first_occurrence_is_reported_as_best(self):
        filler_a = _words("please pay attention now".split())
        first_hit = _words(TARGET.split(), start_at=filler_a[-1].end_seconds)
        filler_b = _words(
            "but I try to focus and keep going".split(), start_at=first_hit[-1].end_seconds
        )
        second_hit = _words(TARGET.split(), start_at=filler_b[-1].end_seconds)
        transcript = _transcript(filler_a + first_hit + filler_b + second_hit)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert result.best.start_seconds == first_hit[0].start_seconds
        assert len(result.candidates) == 2  # both distinct occurrences recorded
        # candidates are chronological: first occurrence, then second.
        assert result.candidates[0].start_seconds < result.candidates[1].start_seconds

    def test_overlapping_stride_one_windows_collapse_to_one_candidate(self):
        # A single occurrence, no repeats -- the stride-1 sliding window
        # will score several overlapping starts highly; only one candidate
        # should survive de-duplication, not five near-duplicates.
        words = _words(("well " + TARGET + " indeed").split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

        assert len(result.candidates) == 1


class TestValidation:
    def test_empty_transcript_raises(self):
        transcript = _transcript([])
        with pytest.raises(MatchingError):
            FuzzyMatcher().match(transcript, TARGET, threshold=80.0)

    def test_empty_target_text_raises(self):
        transcript = _transcript(_words(TARGET.split()))
        with pytest.raises(MatchingError):
            FuzzyMatcher().match(transcript, "", threshold=80.0)

    def test_target_longer_than_transcript_raises(self):
        transcript = _transcript(_words(["hi"]))
        with pytest.raises(MatchingError):
            FuzzyMatcher().match(transcript, TARGET, threshold=80.0)


class TestWindowSizeOverride:
    def test_explicit_window_size_is_respected(self):
        words = _words(TARGET.split())
        transcript = _transcript(words)

        result = FuzzyMatcher().match(transcript, TARGET, threshold=80.0, window_size=5)

        assert result.best.word_end_index - result.best.word_start_index in (4, 5, 6)
