"""Orchestrates all six stages behind their abstract interfaces only.

This is the Dependency Inversion centerpiece: `DialoguePipeline` is
constructed with six interfaces (not concrete classes) and never imports a
single concrete implementation. `main.py` is the only file that wires
interfaces to implementations (the composition root) -- swapping WhisperX
for Vosk, or the fuzzy matcher for something else, never requires touching
this file (Open/Closed).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.audio.base import AudioExtractor
from src.config import PipelineConfig
from src.frame_locator.base import FrameLocator
from src.ingestion.base import VideoDownloader
from src.matching.base import PhraseMatcher
from src.metrics.transcript_metrics import compute_transcript_metrics
from src.output.base import ResultStore
from src.transcription.base import TranscriptionEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRunSummary:
    """What the CLI prints -- a thin view over the full persisted result."""

    timestamp: str
    frame_number: int
    matched_text: str
    match_score: float
    is_uncertain: bool
    uncertainty_reason: str | None
    result_json_path: str
    frame_image_path: str
    total_seconds: float


class DialoguePipeline:
    """Runs the six-stage pipeline described in approach.md §3."""

    def __init__(
        self,
        downloader: VideoDownloader,
        audio_extractor: AudioExtractor,
        transcriber: TranscriptionEngine,
        matcher: PhraseMatcher,
        frame_locator: FrameLocator,
        result_store: ResultStore,
        config: PipelineConfig,
    ) -> None:
        self._downloader = downloader
        self._audio_extractor = audio_extractor
        self._transcriber = transcriber
        self._matcher = matcher
        self._frame_locator = frame_locator
        self._result_store = result_store
        self._config = config

    def run(self) -> PipelineRunSummary:
        start = time.perf_counter()
        cfg = self._config

        logger.info("[1/6] downloading video: %s", cfg.video_url)
        video = self._downloader.download(cfg.video_url, cfg.work_dir)

        logger.info("[2/6] extracting audio")
        audio = self._audio_extractor.extract(video, cfg.work_dir)

        logger.info("[3/6] transcribing (%s, model=%s)", cfg.engine, cfg.whisper_model)
        transcript = self._transcriber.transcribe(audio, cfg.language)

        logger.info("[4/6] computing transcript metrics")
        metrics = compute_transcript_metrics(transcript)

        logger.info("[5/6] matching target phrase: %r", cfg.target_text)
        match = self._matcher.match(
            transcript, cfg.target_text, cfg.match_threshold, cfg.window_size
        )
        if match.is_uncertain:
            logger.warning("match flagged uncertain: %s", match.uncertainty_reason)

        logger.info("[6/6] locating and saving frame at %.3fs", match.best.start_seconds)
        frame = self._frame_locator.locate(video, match.best.start_seconds)
        result_path = self._result_store.save(
            video, cfg.target_text, match, frame, metrics, transcript
        )

        if not cfg.keep_work_files:
            self._cleanup_work_files(video, audio)

        return PipelineRunSummary(
            timestamp=frame.timestamp,
            frame_number=frame.frame_number,
            matched_text=match.best.matched_text,
            match_score=match.best.score,
            is_uncertain=match.is_uncertain,
            uncertainty_reason=match.uncertainty_reason,
            result_json_path=str(result_path),
            frame_image_path=str(result_path.parent.parent / "frames" / f"frame_{frame.frame_number}.png"),
            total_seconds=time.perf_counter() - start,
        )

    def _cleanup_work_files(self, video, audio) -> None:
        for path in (video.file_path, audio.file_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("could not remove work file %s", path, exc_info=True)
