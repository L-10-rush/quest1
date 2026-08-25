"""ffmpeg-CLI-backed AudioExtractor.

Shells out to the `ffmpeg` binary (present on PATH in the Docker image and
required as a native dependency, see README.md) rather than a Python audio
library -- ffmpeg's demuxer/decoder coverage is far broader than any single
Python binding, which matters because stage 1 may hand this stage video in
an arbitrary container/codec depending on the source platform.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import wave
from pathlib import Path

from src.audio.base import AudioAsset, AudioExtractor
from src.exceptions import AudioExtractionError
from src.ingestion.base import VideoMetadata

logger = logging.getLogger(__name__)

# WhisperX (and most ASR models) expect mono 16kHz PCM -- resampling once
# here, rather than inside the transcription engine, keeps that engine
# free of audio-format concerns (Single Responsibility).
TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_CHANNELS = 1

# ffmpeg on a stalled/corrupt input can hang rather than error out --
# bound the subprocess so one bad video can't wedge the whole pipeline.
_FFMPEG_TIMEOUT_SECONDS = 300


class FfmpegAudioExtractor(AudioExtractor):
    """Extracts mono 16kHz WAV audio from a video file via `ffmpeg`."""

    def extract(self, video: VideoMetadata, dest_dir: Path) -> AudioAsset:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"{video.video_id}.wav"

        cmd = [
            "ffmpeg",
            "-y",  # overwrite out_path without prompting (re-runs, retries)
            "-i",
            str(video.file_path),
            "-vn",  # drop the video stream -- only audio is needed downstream
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE_HZ),
            "-f",
            "wav",
            str(out_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise AudioExtractionError(
                "ffmpeg binary not found on PATH -- install it or use the "
                "Docker image, which bundles it"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(
                f"ffmpeg did not finish within {_FFMPEG_TIMEOUT_SECONDS}s "
                f"for {video.file_path}"
            ) from exc

        if result.returncode != 0:
            # ffmpeg's stderr can be very long (codec probing, filter
            # graphs); the last couple KB carries the actual error.
            tail = result.stderr.strip()[-2000:]
            raise AudioExtractionError(
                f"ffmpeg exited {result.returncode} extracting audio from "
                f"{video.file_path}: {tail}"
            )

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise AudioExtractionError(
                f"ffmpeg reported success but produced no audio at {out_path}"
            )

        duration = self._read_wav_duration(out_path)
        logger.info(
            "extracted audio: %s (%.1fs, %dHz mono)", out_path, duration, TARGET_SAMPLE_RATE_HZ
        )
        return AudioAsset(
            file_path=out_path,
            sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
            channels=TARGET_CHANNELS,
            duration_seconds=duration,
        )

    @staticmethod
    def _read_wav_duration(path: Path) -> float:
        """Read duration from the WAV header itself rather than trusting
        ffmpeg's stderr -- avoids parsing ffmpeg's human-oriented log
        format for something the file already states unambiguously."""
        with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate == 0:
                return 0.0
            return wav_file.getnframes() / frame_rate
