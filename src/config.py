"""Run configuration: one immutable object threaded through the whole pipeline.

Every tunable knob (model size, threshold, directories, ...) lives here so
that `pipeline.py` and the stage implementations never read `os.environ` or
`sys.argv` directly -- they only ever see a `PipelineConfig`. That keeps the
stages testable with plain Python objects instead of environment mocking.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LANGUAGE = "en"
DEFAULT_MATCH_THRESHOLD = 80.0
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_ENGINE = "whisperx"
DEFAULT_DEVICE = "cpu"


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single pipeline run.

    Frozen so a config can be safely shared across stage objects without a
    stage accidentally mutating it mid-run.
    """

    video_url: str
    target_text: str
    language: str = DEFAULT_LANGUAGE
    engine: str = DEFAULT_ENGINE  # "whisperx" | "vosk"
    whisper_model: str = DEFAULT_WHISPER_MODEL
    device: str = DEFAULT_DEVICE  # "cpu" | "cuda"
    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    window_size: int | None = None  # None => matcher picks a sane default
    work_dir: Path = field(default_factory=lambda: Path("work"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    keep_work_files: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.video_url.strip():
            raise ValueError("video_url must not be empty")
        if not self.target_text.strip():
            raise ValueError("target_text must not be empty")
        if not 0 <= self.match_threshold <= 100:
            raise ValueError("match_threshold must be between 0 and 100")
        if self.engine not in ("whisperx", "vosk"):
            raise ValueError(f"unknown engine: {self.engine!r}")


def _env_default(name: str, fallback: str) -> str:
    """Read an env var with a fallback, used only for argparse defaults so
    a `.env` (see `.env.example`) can override defaults without code
    changes -- CLI flags still win over both."""
    return os.environ.get(name, fallback)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-dialogue-finder",
        description=(
            "Find the exact video frame where a spoken line of dialogue "
            "occurs, using audio transcription + fuzzy phrase matching."
        ),
    )
    parser.add_argument("--url", required=True, help="Source video URL (e.g. YouTube, ok.ru).")
    parser.add_argument("--text", required=True, help="Target dialogue text to locate.")
    parser.add_argument(
        "--language",
        default=_env_default("LANGUAGE", DEFAULT_LANGUAGE),
        help=f"ISO-639-1 language code for ASR (default: {DEFAULT_LANGUAGE}).",
    )
    parser.add_argument(
        "--engine",
        choices=("whisperx", "vosk"),
        default=_env_default("ENGINE", DEFAULT_ENGINE),
        help="Transcription engine to use (default: whisperx).",
    )
    parser.add_argument(
        "--whisper-model",
        default=_env_default("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        help="WhisperX model size: tiny|base|small|medium|large-v3 (default: small).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=_env_default("DEVICE", DEFAULT_DEVICE),
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=float(_env_default("MATCH_THRESHOLD", str(DEFAULT_MATCH_THRESHOLD))),
        help="Minimum fuzzy-match score [0-100] to be considered confident (default: 80).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Sliding-window width in words. Default: auto (len(target_text.split())).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(_env_default("WORK_DIR", "work")),
        help="Scratch directory for downloaded video / extracted audio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_env_default("OUTPUT_DIR", "output")),
        help="Directory where per-video result.json + frames/ are written.",
    )
    parser.add_argument(
        "--keep-work-files",
        action="store_true",
        help="Do not delete downloaded video / extracted audio after the run.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    return parser


def config_from_args(argv: list[str] | None = None) -> PipelineConfig:
    """Parse CLI args (or `argv`, for tests) into a `PipelineConfig`."""
    args = build_arg_parser().parse_args(argv)
    return PipelineConfig(
        video_url=args.url,
        target_text=args.text,
        language=args.language,
        engine=args.engine,
        whisper_model=args.whisper_model,
        device=args.device,
        match_threshold=args.match_threshold,
        window_size=args.window_size,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        keep_work_files=args.keep_work_files,
        verbose=args.verbose,
    )
