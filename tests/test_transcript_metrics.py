from src.metrics.transcript_metrics import compute_transcript_metrics
from src.transcription.base import TranscriptResult, Word


def _transcript(*specs) -> TranscriptResult:
    """specs: list of (text, start, end, confidence)."""
    words = tuple(Word(text=t, start_seconds=s, end_seconds=e, confidence=c) for t, s, e, c in specs)
    return TranscriptResult(words=words, language="en", engine_name="test", model_name="test")


class TestComputeTranscriptMetrics:
    def test_empty_transcript(self):
        metrics = compute_transcript_metrics(_transcript())
        assert metrics.total_words == 0
        assert metrics.unique_words == 0
        assert metrics.word_frequencies == {}
        assert metrics.words_per_minute == 0.0

    def test_word_counts_and_frequencies(self):
        transcript = _transcript(
            ("My", 0.0, 0.2, 0.9),
            ("mind", 0.2, 0.5, 0.95),
            ("rebels", 0.5, 0.9, 0.8),
            ("at", 0.9, 1.0, 0.99),
            ("stagnation", 1.0, 1.6, 0.85),
        )
        metrics = compute_transcript_metrics(transcript)
        assert metrics.total_words == 5
        assert metrics.unique_words == 5
        assert metrics.word_frequencies["mind"] == 1

    def test_punctuation_and_case_normalized_into_same_bucket(self):
        transcript = _transcript(
            ("Stagnation,", 0.0, 0.5, 0.9),
            ("stagnation", 0.5, 1.0, 0.9),
            ("STAGNATION!", 1.0, 1.5, 0.9),
        )
        metrics = compute_transcript_metrics(transcript)
        assert metrics.unique_words == 1
        assert metrics.word_frequencies["stagnation"] == 3

    def test_confidence_stats(self):
        transcript = _transcript(
            ("a", 0.0, 0.1, 0.5),
            ("b", 0.1, 0.2, 1.0),
            ("c", 0.2, 0.3, 0.0),
        )
        metrics = compute_transcript_metrics(transcript)
        assert metrics.min_confidence == 0.0
        assert metrics.max_confidence == 1.0
        assert round(metrics.average_confidence, 4) == round(1.5 / 3, 4)

    def test_words_per_minute(self):
        # 3 words spanning exactly 30 seconds -> 6 words/minute
        transcript = _transcript(
            ("a", 0.0, 1.0, 1.0),
            ("b", 5.0, 6.0, 1.0),
            ("c", 29.0, 30.0, 1.0),
        )
        metrics = compute_transcript_metrics(transcript)
        assert metrics.transcript_duration_seconds == 30.0
        assert metrics.words_per_minute == 6.0
