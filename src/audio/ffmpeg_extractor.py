"""ffmpeg-CLI-backed AudioExtractor.

STATUS: scaffold only -- `extract()` is intentionally left unimplemented.
The ffmpeg invocation needs a real downloaded video file to validate exit
codes / stderr parsing against, so it's left for you. Wiring and the
expected command are documented below.
"""

from __future__ import annotations

import logging
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


class FfmpegAudioExtractor(AudioExtractor):
    """Shells out to the `ffmpeg` binary (already on PATH in the Docker image)."""

    def extract(self, video: VideoMetadata, dest_dir: Path) -> AudioAsset:
        """Extract mono 16kHz WAV audio from `video.file_path`.

        TODO (not implemented -- fill this in):

        1. `dest_dir.mkdir(parents=True, exist_ok=True)`
           `out_path = dest_dir / f"{video.video_id}.wav"`
        2. Run ffmpeg via subprocess, e.g.:
               import subprocess
               cmd = [
                   "ffmpeg", "-y", "-i", str(video.file_path),
                   "-vn",                                   # drop video stream
                   "-ac", str(TARGET_CHANNELS),
                   "-ar", str(TARGET_SAMPLE_RATE_HZ),
                   "-f", "wav", str(out_path),
               ]
               result = subprocess.run(cmd, capture_output=True, text=True)
        3. If `result.returncode != 0`: raise
           `AudioExtractionError(result.stderr)`.
        4. Verify `out_path.exists()` and is non-empty; raise
           `AudioExtractionError(...)` if not (ffmpeg can exit 0 with a
           truncated file on some malformed inputs).
        5. Compute `duration_seconds` -- either parse it from ffmpeg's
           stderr, or reopen the WAV with the stdlib `wave` module and use
           `frames / framerate`.
        6. Return `AudioAsset(file_path=out_path,
           sample_rate_hz=TARGET_SAMPLE_RATE_HZ, channels=TARGET_CHANNELS,
           duration_seconds=duration_seconds)`.
        """
        raise NotImplementedError(
            "FfmpegAudioExtractor.extract() is a scaffold -- implement steps "
            "1-6 in the docstring above and validate against a real video file."
        )
