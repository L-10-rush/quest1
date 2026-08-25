"""Unit tests for FfmpegAudioExtractor against a real `ffmpeg` binary.

Uses ffmpeg's own `lavfi` source filters to synthesize a tiny video+audio
fixture on the fly (no network, no committed binary test asset) -- the
same "exercise the real tool against a small deterministic input" approach
`test_opencv_locator.py` uses for OpenCV. `ffmpeg` itself is a required
runtime dependency of this project (README.md, Dockerfile), so testing
against the real binary is testing what will actually run in production,
not a stand-in for it.
"""

import shutil
import wave
from pathlib import Path

import pytest

from src.audio.base import AudioAsset
from src.audio.ffmpeg_extractor import TARGET_SAMPLE_RATE_HZ, FfmpegAudioExtractor
from src.exceptions import AudioExtractionError
from src.ingestion.base import VideoMetadata

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg binary not available on PATH"
)


def _synthesize_video_with_audio(
    path: Path, duration: float = 1.0, frequency: int = 440, has_audio: bool = True
) -> None:
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=64x64:rate=10",
    ]
    if has_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={duration}"]
        cmd += ["-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path)]
    else:
        cmd += ["-c:v", "mpeg4", "-an", str(path)]
    subprocess.run(cmd, capture_output=True, check=True)


def _video_metadata(file_path: Path) -> VideoMetadata:
    return VideoMetadata(
        video_id="test_audio",
        source_url="https://example.com/test",
        file_path=file_path,
        title="synthetic test video",
        duration_seconds=1.0,
        fps=10.0,
        width=64,
        height=64,
    )


class TestFfmpegAudioExtractor:
    def test_extracts_mono_16khz_wav(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        _synthesize_video_with_audio(video_path, duration=1.0)
        video = _video_metadata(video_path)

        asset = FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)

        assert isinstance(asset, AudioAsset)
        assert asset.file_path.exists()
        assert asset.sample_rate_hz == TARGET_SAMPLE_RATE_HZ
        assert asset.channels == 1

    def test_output_file_is_actually_mono_16khz(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        _synthesize_video_with_audio(video_path, duration=1.0)
        video = _video_metadata(video_path)

        asset = FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)

        with wave.open(str(asset.file_path), "rb") as wav_file:
            assert wav_file.getframerate() == TARGET_SAMPLE_RATE_HZ
            assert wav_file.getnchannels() == 1

    def test_duration_matches_source(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        _synthesize_video_with_audio(video_path, duration=2.0)
        video = _video_metadata(video_path)

        asset = FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)

        assert asset.duration_seconds == pytest.approx(2.0, abs=0.1)

    def test_output_named_after_video_id(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        _synthesize_video_with_audio(video_path, duration=1.0)
        video = _video_metadata(video_path)

        asset = FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)

        assert asset.file_path.name == f"{video.video_id}.wav"

    def test_creates_dest_dir_if_missing(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        _synthesize_video_with_audio(video_path, duration=1.0)
        video = _video_metadata(video_path)
        nested_dest = tmp_path / "nested" / "work"

        asset = FfmpegAudioExtractor().extract(video, dest_dir=nested_dest)

        assert asset.file_path.exists()

    def test_video_without_an_audio_track_raises(self, tmp_path: Path):
        video_path = tmp_path / "silent.mp4"
        _synthesize_video_with_audio(video_path, duration=1.0, has_audio=False)
        video = _video_metadata(video_path)

        with pytest.raises(AudioExtractionError):
            FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)

    def test_missing_source_file_raises_audio_extraction_error(self, tmp_path: Path):
        video = _video_metadata(tmp_path / "does_not_exist.mp4")

        with pytest.raises(AudioExtractionError):
            FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)

    def test_ffmpeg_not_on_path_raises_audio_extraction_error(self, tmp_path: Path, monkeypatch):
        video_path = tmp_path / "video.mp4"
        _synthesize_video_with_audio(video_path, duration=1.0)
        video = _video_metadata(video_path)
        monkeypatch.setenv("PATH", "")  # make the `ffmpeg` binary unresolvable

        with pytest.raises(AudioExtractionError, match="ffmpeg binary not found"):
            FfmpegAudioExtractor().extract(video, dest_dir=tmp_path)
