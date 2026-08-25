import json
from pathlib import Path

import numpy as np
import pytest

from src.frame_locator.base import FrameResult
from src.ingestion.base import VideoMetadata
from src.matching.base import MatchCandidate, MatchResult
from src.metrics.transcript_metrics import TranscriptMetrics
from src.output.json_store import JsonResultStore
from src.transcription.base import DialogueSegment, TranscriptResult, Word


@pytest.fixture()
def video() -> VideoMetadata:
    return VideoMetadata(
        video_id="248244667877",
        source_url="https://ok.ru/video/248244667877",
        file_path=Path("work/248244667877.mp4"),
        title="test video",
        duration_seconds=120.0,
        fps=25.0,
        width=1280,
        height=720,
    )


@pytest.fixture()
def match() -> MatchResult:
    candidate = MatchCandidate(
        matched_text="my mind rebels at stagnation",
        start_seconds=42.0,
        end_seconds=44.5,
        score=96.5,
        word_start_index=100,
        word_end_index=105,
    )
    return MatchResult(
        best=candidate, candidates=(candidate,), is_uncertain=False, uncertainty_reason=None
    )


@pytest.fixture()
def frame() -> FrameResult:
    return FrameResult(
        frame_number=1050,
        timestamp="00:00:42.000",
        timestamp_seconds=42.0,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


@pytest.fixture()
def metrics() -> TranscriptMetrics:
    return TranscriptMetrics(
        total_words=200,
        unique_words=120,
        word_frequencies={"the": 10, "mind": 3},
        average_confidence=0.91,
        min_confidence=0.4,
        max_confidence=1.0,
        transcript_duration_seconds=118.0,
        words_per_minute=101.7,
    )


@pytest.fixture()
def transcript() -> TranscriptResult:
    words = (Word(text="hi", start_seconds=0.0, end_seconds=0.3, confidence=0.9),)
    segments = (
        DialogueSegment(
            text="Well, hello there.", start_seconds=0.0, end_seconds=1.2, confidence=0.88
        ),
        DialogueSegment(
            text="My mind rebels at stagnation.",
            start_seconds=42.0,
            end_seconds=44.5,
            confidence=0.95,
        ),
    )
    return TranscriptResult(
        words=words, segments=segments, language="en", engine_name="whisperx", model_name="small"
    )


class TestJsonResultStore:
    def test_writes_result_json_and_frame_image(
        self, tmp_path: Path, video, match, frame, metrics, transcript
    ):
        store = JsonResultStore(output_dir=tmp_path)
        result_path = store.save(
            video, "My mind rebels at stagnation", match, frame, metrics, transcript
        )

        assert result_path.exists()
        image_path = tmp_path / video.video_id / "frames" / f"frame_{frame.frame_number}.png"
        assert image_path.exists()

        payload = json.loads(result_path.read_text())
        assert payload["result"]["timestamp"] == "00:00:42.000"
        assert payload["result"]["frame_number"] == 1050
        assert payload["result"]["matched_text"] == "my mind rebels at stagnation"
        assert payload["query"]["target_text"] == "My mind rebels at stagnation"
        assert payload["transcript_metrics"]["total_words"] == 200
        assert payload["transcript_metrics"]["word_frequencies"]["the"] == 10

    def test_writes_full_transcript_of_every_dialogue_line(
        self, tmp_path: Path, video, match, frame, metrics, transcript
    ):
        store = JsonResultStore(output_dir=tmp_path)
        result_path = store.save(video, "anything", match, frame, metrics, transcript)

        payload = json.loads(result_path.read_text())
        assert len(payload["transcript"]) == 2
        assert payload["transcript"][0]["text"] == "Well, hello there."
        assert payload["transcript"][0]["start_timestamp"] == "00:00:00.000"
        assert payload["transcript"][1]["text"] == "My mind rebels at stagnation."
        assert payload["transcript"][1]["start_seconds"] == 42.0
        assert payload["transcript"][1]["confidence"] == 0.95

    def test_video_meta_written_once_and_reused_across_runs(
        self, tmp_path: Path, video, match, frame, metrics, transcript
    ):
        store = JsonResultStore(output_dir=tmp_path)
        store.save(video, "first query", match, frame, metrics, transcript)
        meta_path = tmp_path / video.video_id / f"{video.video_id}.meta.json"
        first_mtime = meta_path.stat().st_mtime_ns

        store.save(video, "second query", match, frame, metrics, transcript)
        assert meta_path.stat().st_mtime_ns == first_mtime

    def test_rejects_match_result_with_no_best(
        self, tmp_path: Path, video, frame, metrics, transcript
    ):
        from src.exceptions import ResultPersistenceError

        empty_match = MatchResult(
            best=None, candidates=(), is_uncertain=True, uncertainty_reason="no words"
        )
        store = JsonResultStore(output_dir=tmp_path)
        with pytest.raises(ResultPersistenceError):
            store.save(video, "anything", empty_match, frame, metrics, transcript)
