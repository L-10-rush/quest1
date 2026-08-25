"""Integration test of DialoguePipeline.run() against fake stage
implementations -- no real network, model, or media file involved.

Sprint 3 (robustness): proves the six stages actually wire together in the
right order and pass the right data at each hand-off (Dependency
Inversion in practice, not just in theory), that an uncertain match still
produces a full result instead of a silent/partial failure, that
`--language` reaches the transcription stage untouched by pipeline code,
and that work-file cleanup respects `--keep-work-files`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.base import AudioAsset, AudioExtractor
from src.config import PipelineConfig
from src.frame_locator.base import FrameLocator, FrameResult
from src.ingestion.base import VideoDownloader, VideoMetadata
from src.matching.base import MatchCandidate, MatchResult, PhraseMatcher
from src.output.base import ResultStore
from src.pipeline import DialoguePipeline
from src.transcription.base import TranscriptionEngine, TranscriptResult, Word


class FakeDownloader(VideoDownloader):
    def __init__(self, video: VideoMetadata):
        self.video = video
        self.calls: list[tuple[str, Path]] = []

    def download(self, url: str, dest_dir: Path) -> VideoMetadata:
        self.calls.append((url, dest_dir))
        return self.video


class FakeAudioExtractor(AudioExtractor):
    def __init__(self, audio: AudioAsset):
        self.audio = audio
        self.calls: list[tuple[VideoMetadata, Path]] = []

    def extract(self, video: VideoMetadata, dest_dir: Path) -> AudioAsset:
        self.calls.append((video, dest_dir))
        return self.audio


class FakeTranscriber(TranscriptionEngine):
    def __init__(self, transcript: TranscriptResult):
        self.transcript = transcript
        self.calls: list[tuple[AudioAsset, str]] = []

    def transcribe(self, audio: AudioAsset, language: str) -> TranscriptResult:
        self.calls.append((audio, language))
        return self.transcript


class FakeMatcher(PhraseMatcher):
    def __init__(self, result: MatchResult):
        self.result = result
        self.calls: list[tuple[TranscriptResult, str, float, int | None]] = []

    def match(self, transcript, target_text, threshold, window_size=None) -> MatchResult:
        self.calls.append((transcript, target_text, threshold, window_size))
        return self.result


class FakeFrameLocator(FrameLocator):
    def __init__(self, frame: FrameResult):
        self.frame = frame
        self.calls: list[tuple[VideoMetadata, float]] = []

    def locate(self, video: VideoMetadata, timestamp_seconds: float) -> FrameResult:
        self.calls.append((video, timestamp_seconds))
        return self.frame


class FakeResultStore(ResultStore):
    def __init__(self, result_path: Path):
        self.result_path = result_path
        self.calls: list[tuple] = []

    def save(self, video, target_text, match, frame, metrics, transcript) -> Path:
        self.calls.append((video, target_text, match, frame, metrics, transcript))
        return self.result_path


@pytest.fixture()
def transcript() -> TranscriptResult:
    words = tuple(
        Word(text=t, start_seconds=i * 0.3, end_seconds=i * 0.3 + 0.3, confidence=0.9)
        for i, t in enumerate("My mind rebels at stagnation".split())
    )
    return TranscriptResult(words=words, language="en", engine_name="fake", model_name="fake")


@pytest.fixture()
def confident_match(transcript: TranscriptResult) -> MatchResult:
    candidate = MatchCandidate(
        matched_text="My mind rebels at stagnation",
        start_seconds=0.0,
        end_seconds=1.5,
        score=100.0,
        word_start_index=0,
        word_end_index=5,
    )
    return MatchResult(best=candidate, candidates=(candidate,), is_uncertain=False, uncertainty_reason=None)


def _build_pipeline(tmp_path: Path, config: PipelineConfig, match_result: MatchResult):
    video_file = tmp_path / "video.mp4"
    audio_file = tmp_path / "audio.wav"
    video_file.write_bytes(b"fake video")
    audio_file.write_bytes(b"fake audio")

    video = VideoMetadata(
        video_id="248244667877",
        source_url=config.video_url,
        file_path=video_file,
        title="test video",
        duration_seconds=120.0,
        fps=25.0,
        width=1280,
        height=720,
        sequence_id=1,
    )
    audio = AudioAsset(file_path=audio_file, sample_rate_hz=16_000, channels=1, duration_seconds=120.0)
    words = tuple(
        Word(text=t, start_seconds=i * 0.3, end_seconds=i * 0.3 + 0.3, confidence=0.9)
        for i, t in enumerate("My mind rebels at stagnation".split())
    )
    transcript = TranscriptResult(words=words, language=config.language, engine_name="fake", model_name="fake")
    frame = FrameResult(
        frame_number=105,
        timestamp="00:00:01.500",
        timestamp_seconds=1.5,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
    )
    result_path = tmp_path / "output" / "248244667877" / "results" / "result_105.json"

    downloader = FakeDownloader(video)
    audio_extractor = FakeAudioExtractor(audio)
    transcriber = FakeTranscriber(transcript)
    matcher = FakeMatcher(match_result)
    frame_locator = FakeFrameLocator(frame)
    result_store = FakeResultStore(result_path)

    pipeline = DialoguePipeline(
        downloader=downloader,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        matcher=matcher,
        frame_locator=frame_locator,
        result_store=result_store,
        config=config,
    )
    fakes = {
        "downloader": downloader,
        "audio_extractor": audio_extractor,
        "transcriber": transcriber,
        "matcher": matcher,
        "frame_locator": frame_locator,
        "result_store": result_store,
        "video_file": video_file,
        "audio_file": audio_file,
    }
    return pipeline, fakes


class TestDialoguePipelineOrchestration:
    def test_stages_are_called_in_order_with_correct_data_handoff(
        self, tmp_path: Path, confident_match: MatchResult
    ):
        config = PipelineConfig(
            video_url="https://ok.ru/video/248244667877",
            target_text="My mind rebels at stagnation",
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            keep_work_files=True,  # so we can still inspect the work files after run()
        )
        pipeline, fakes = _build_pipeline(tmp_path, config, confident_match)

        summary = pipeline.run()

        assert fakes["downloader"].calls == [(config.video_url, config.work_dir)]
        video = fakes["downloader"].video
        assert fakes["audio_extractor"].calls == [(video, config.work_dir)]
        audio = fakes["audio_extractor"].audio
        assert fakes["transcriber"].calls == [(audio, config.language)]
        assert fakes["matcher"].calls[0][1] == config.target_text
        assert fakes["matcher"].calls[0][2] == config.match_threshold
        # frame locator is called with the MATCHER's chosen start time, not
        # some independently recomputed value -- proves the hand-off.
        assert fakes["frame_locator"].calls == [(video, confident_match.best.start_seconds)]

        assert summary.timestamp == "00:00:01.500"
        assert summary.frame_number == 105
        assert summary.matched_text == "My mind rebels at stagnation"
        assert not summary.is_uncertain

    def test_language_flows_to_transcriber_untouched_by_pipeline_code(
        self, tmp_path: Path, confident_match: MatchResult
    ):
        """Sprint 3 exit criterion: --language is swappable without
        touching pipeline.py -- proven by driving two different languages
        through the same DialoguePipeline construction path."""
        for language in ("en", "ja", "es"):
            config = PipelineConfig(
                video_url="https://ok.ru/video/248244667877",
                target_text="hello",
                language=language,
                work_dir=tmp_path / language,
                output_dir=tmp_path / language / "output",
                keep_work_files=True,
            )
            (tmp_path / language).mkdir()
            pipeline, fakes = _build_pipeline(tmp_path / language, config, confident_match)

            pipeline.run()

            assert fakes["transcriber"].calls[0][1] == language

    def test_uncertain_match_still_produces_a_full_result(self, tmp_path: Path, transcript):
        """Never a silent failure: even a low-confidence/no-match result
        flows all the way through to a saved result and a CLI summary."""
        uncertain_candidate = MatchCandidate(
            matched_text="some other words entirely",
            start_seconds=3.0,
            end_seconds=4.0,
            score=42.0,
            word_start_index=0,
            word_end_index=4,
        )
        uncertain_match = MatchResult(
            best=uncertain_candidate,
            candidates=(),
            is_uncertain=True,
            uncertainty_reason="best match scored 42.0, below threshold 80.0",
        )
        config = PipelineConfig(
            video_url="https://ok.ru/video/248244667877",
            target_text="My mind rebels at stagnation",
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            keep_work_files=True,
        )
        pipeline, fakes = _build_pipeline(tmp_path, config, uncertain_match)

        summary = pipeline.run()

        assert summary.is_uncertain
        assert summary.uncertainty_reason == uncertain_match.uncertainty_reason
        assert summary.matched_text == "some other words entirely"
        # the frame locator and result store still ran -- nothing short-circuited
        assert len(fakes["frame_locator"].calls) == 1
        assert len(fakes["result_store"].calls) == 1

    def test_uncertain_match_logs_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        uncertain_candidate = MatchCandidate(
            matched_text="x", start_seconds=0.0, end_seconds=0.3, score=10.0,
            word_start_index=0, word_end_index=1,
        )
        uncertain_match = MatchResult(
            best=uncertain_candidate, candidates=(), is_uncertain=True,
            uncertainty_reason="no good match",
        )
        config = PipelineConfig(
            video_url="https://ok.ru/video/248244667877",
            target_text="anything",
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            keep_work_files=True,
        )
        pipeline, _ = _build_pipeline(tmp_path, config, uncertain_match)

        with caplog.at_level("WARNING"):
            pipeline.run()

        assert any("uncertain" in record.message for record in caplog.records)

    def test_work_files_deleted_by_default(self, tmp_path: Path, confident_match: MatchResult):
        config = PipelineConfig(
            video_url="https://ok.ru/video/248244667877",
            target_text="My mind rebels at stagnation",
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            keep_work_files=False,
        )
        pipeline, fakes = _build_pipeline(tmp_path, config, confident_match)

        pipeline.run()

        assert not fakes["video_file"].exists()
        assert not fakes["audio_file"].exists()

    def test_work_files_kept_when_flag_set(self, tmp_path: Path, confident_match: MatchResult):
        config = PipelineConfig(
            video_url="https://ok.ru/video/248244667877",
            target_text="My mind rebels at stagnation",
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            keep_work_files=True,
        )
        pipeline, fakes = _build_pipeline(tmp_path, config, confident_match)

        pipeline.run()

        assert fakes["video_file"].exists()
        assert fakes["audio_file"].exists()

    def test_summary_reports_elapsed_time(self, tmp_path: Path, confident_match: MatchResult):
        config = PipelineConfig(
            video_url="https://ok.ru/video/248244667877",
            target_text="My mind rebels at stagnation",
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            keep_work_files=True,
        )
        pipeline, _ = _build_pipeline(tmp_path, config, confident_match)

        summary = pipeline.run()

        assert summary.total_seconds >= 0.0
