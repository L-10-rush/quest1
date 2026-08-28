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
from urllib.parse import urlparse

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
    # None => no phrase given on the CLI, so main.py starts an interactive
    # session instead of a single-shot search (see main.py).
    target_text: str | None = None
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
    # Stage 6: whether to verify the speaker was visibly on camera, not
    # just that the line was said somewhere in the audio (see
    # screen_presence/base.py). On by default -- it's what answers the
    # literal "on-screen dialogue" reading of the problem statement.
    verify_screen_presence: bool = True
    # Whether to also extract preview frames for the other top-scoring
    # candidates, not just the winning match (see pipeline.py's
    # _MAX_CANDIDATE_PREVIEWS). Off by default -- the CLI has no way to
    # display these, so this only matters to a caller that renders images
    # itself, e.g. the optional Streamlit UI (src/webapp/app.py), which
    # sets it explicitly per its own "extract candidate previews" toggle.
    extract_candidate_previews: bool = False
    # Plain-text, human-readable record of every search against this video,
    # appended to output/<video_id>/session.log after each one -- separate
    # from the per-search result_<frame>.json (see output/json_store.py).
    save_session_log: bool = True

    def __post_init__(self) -> None:
        stripped_url = self.video_url.strip()
        if not stripped_url:
            raise ValueError("video_url must not be empty")
        # Cheap, no-network sanity check -- catches a typo'd/garbage URL
        # (missing scheme, no host, wrong protocol) immediately at
        # construction time rather than after a slow, wrapped failure deep
        # inside YtDlpDownloader. A URL that parses fine here can still
        # fail at download time (host unreachable, video not found) --
        # that's handled separately in YtDlpDownloader.download().
        parsed = urlparse(stripped_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "video_url must be a valid http(s) URL (e.g. "
                f"https://example.com/video), got: {self.video_url!r}"
            )
        if self.target_text is not None and not self.target_text.strip():
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
    parser.add_argument(
        "--text",
        default=None,
        help=(
            "Target dialogue text to locate. If omitted, the video is downloaded and "
            "transcribed once, then an interactive session prompts for dialogue lines "
            "to search one at a time until you exit."
        ),
    )
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
    parser.add_argument(
        "--no-screen-verification",
        action="store_false",
        dest="verify_screen_presence",
        default=True,
        help=(
            "Skip on-screen speaker verification (stage 6) -- only report where "
            "the line was said, not whether the speaker was visibly on camera."
        ),
    )
    parser.add_argument(
        "--no-session-log",
        action="store_false",
        dest="save_session_log",
        default=True,
        help=(
            "Don't append a plain-text record of each search to "
            "output/<video_id>/session.log."
        ),
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
        verify_screen_presence=args.verify_screen_presence,
        save_session_log=args.save_session_log,
    )
