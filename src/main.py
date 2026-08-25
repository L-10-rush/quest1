"""CLI entrypoint and composition root.

The ONLY file in the project that imports concrete implementations
(`YtDlpDownloader`, `FfmpegAudioExtractor`, `WhisperXEngine`/`VoskEngine`,
`FuzzyMatcher`, `OpenCvFrameLocator`, `JsonResultStore`) and wires them
into `DialoguePipeline`. Every other file depends only on the `base.py`
interfaces -- this is the Dependency Inversion boundary made concrete.

Run via `python -m src.main --url ... --text ...` (see README.md), or
through the Docker image where this is the container ENTRYPOINT.
"""

from __future__ import annotations

import logging
import sys

from src.audio.ffmpeg_extractor import FfmpegAudioExtractor
from src.config import PipelineConfig, config_from_args
from src.exceptions import PipelineError
from src.frame_locator.opencv_locator import OpenCvFrameLocator
from src.ingestion.ytdlp_downloader import YtDlpDownloader
from src.logging_config import configure_logging
from src.matching.fuzzy_matcher import FuzzyMatcher
from src.output.json_store import JsonResultStore
from src.pipeline import DialoguePipeline, PipelineRunSummary
from src.transcription.base import TranscriptionEngine
from src.transcription.vosk_engine import VoskEngine
from src.transcription.whisperx_engine import WhisperXEngine

logger = logging.getLogger(__name__)


def _build_transcriber(config: PipelineConfig) -> TranscriptionEngine:
    """The one place that decides WhisperX vs. Vosk -- an Open/Closed seam:
    adding a third engine means adding one `elif`, not touching pipeline.py."""
    if config.engine == "whisperx":
        return WhisperXEngine(model_size=config.whisper_model, device=config.device)
    if config.engine == "vosk":
        return VoskEngine(model_path="models/vosk")
    raise ValueError(f"unknown engine: {config.engine!r}")  # unreachable: config validates this


def build_pipeline(config: PipelineConfig) -> DialoguePipeline:
    return DialoguePipeline(
        downloader=YtDlpDownloader(registry_path=config.output_dir / "registry.db"),
        audio_extractor=FfmpegAudioExtractor(),
        transcriber=_build_transcriber(config),
        matcher=FuzzyMatcher(),
        frame_locator=OpenCvFrameLocator(),
        result_store=JsonResultStore(config.output_dir),
        config=config,
    )


def _print_summary(summary: PipelineRunSummary) -> None:
    print(f"Timestamp : {summary.timestamp}")
    print(f"Frame     : {summary.frame_number}")
    print(f"Text      : \"{summary.matched_text}\"")
    print(f"Score     : {summary.match_score:.1f}")
    if summary.is_uncertain:
        print(f"Warning   : result flagged UNCERTAIN -- {summary.uncertainty_reason}")
    print(f"Image     : {summary.frame_image_path}")
    print(f"JSON      : {summary.result_json_path}")
    print(f"Elapsed   : {summary.total_seconds:.1f}s")


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(argv)
    configure_logging(config.verbose)

    try:
        pipeline = build_pipeline(config)
        summary = pipeline.run()
    except PipelineError as exc:
        logger.error("pipeline failed: %s", exc)
        return 1
    except NotImplementedError as exc:
        # Expected while the scaffold's TODO stages are unimplemented --
        # surfaced as a clean CLI error instead of a raw traceback.
        logger.error("not yet implemented: %s", exc)
        return 2

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
