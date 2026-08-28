"""Unit tests for YtDlpDownloader's orchestration: registry integration,
sequenced naming, caching, and error wrapping.

Network/yt-dlp/OpenCV calls are mocked out at the `_run_ytdlp` / `_probe`
seam (monkeypatch) -- these tests verify OUR orchestration logic, not
yt-dlp's or OpenCV's own behaviour, so they run fast with no network and
no real media file. A separate, manual/integration check against the real
assignment URL is documented in approach.md, not part of this fast suite.
"""

from pathlib import Path

import pytest

from src.exceptions import DownloadError
from src.ingestion.registry import VideoRegistry
from src.ingestion.ytdlp_downloader import YtDlpDownloader

URL = "https://ok.ru/video/248244667877"


def _stub_run_ytdlp(monkeypatch, downloader: YtDlpDownloader, dest_dir: Path, call_log: list):
    """Replace the real yt-dlp call with one that just writes an empty file
    and records how many times it was invoked (to prove caching works)."""

    def fake_run_ytdlp(self, url, dest_dir, filename_stem):
        call_log.append(url)
        file_path = dest_dir / f"{filename_stem}.mp4"
        file_path.write_bytes(b"fake mp4 bytes")
        return file_path, "Stub Title"

    monkeypatch.setattr(YtDlpDownloader, "_run_ytdlp", fake_run_ytdlp)


def _stub_probe(monkeypatch):
    def fake_probe(self, file_path):
        return 25.0, 10.0, 1280, 720  # fps, duration, width, height

    monkeypatch.setattr(YtDlpDownloader, "_probe", fake_probe)


@pytest.fixture()
def registry(tmp_path: Path) -> VideoRegistry:
    return VideoRegistry(tmp_path / "registry.db")


class TestDownload:
    def test_downloads_and_returns_metadata(self, tmp_path, monkeypatch, registry):
        call_log: list = []
        downloader = YtDlpDownloader(registry=registry)
        _stub_run_ytdlp(monkeypatch, downloader, tmp_path, call_log)
        _stub_probe(monkeypatch)

        metadata = downloader.download(URL, dest_dir=tmp_path)

        assert metadata.video_id == "248244667877"
        assert metadata.title == "Stub Title"
        assert metadata.fps == 25.0
        assert metadata.duration_seconds == 10.0
        assert metadata.file_path.exists()
        assert call_log == [URL]

    def test_filename_uses_sequenced_stem(self, tmp_path, monkeypatch, registry):
        downloader = YtDlpDownloader(registry=registry)
        _stub_run_ytdlp(monkeypatch, downloader, tmp_path, [])
        _stub_probe(monkeypatch)

        metadata = downloader.download(URL, dest_dir=tmp_path)

        assert metadata.sequence_id == 1
        assert metadata.file_path.name == "000001_248244667877.mp4"

    def test_second_call_for_same_url_is_a_cache_hit(self, tmp_path, monkeypatch, registry):
        call_log: list = []
        downloader = YtDlpDownloader(registry=registry)
        _stub_run_ytdlp(monkeypatch, downloader, tmp_path, call_log)
        _stub_probe(monkeypatch)

        first = downloader.download(URL, dest_dir=tmp_path)
        second = downloader.download(URL, dest_dir=tmp_path)

        assert call_log == [URL]  # yt-dlp only invoked once
        assert first == second

    def test_different_urls_get_increasing_sequence_ids(self, tmp_path, monkeypatch, registry):
        downloader = YtDlpDownloader(registry=registry)
        _stub_run_ytdlp(monkeypatch, downloader, tmp_path, [])
        _stub_probe(monkeypatch)

        first = downloader.download("https://ok.ru/video/111", dest_dir=tmp_path)
        second = downloader.download("https://ok.ru/video/222", dest_dir=tmp_path)

        assert first.sequence_id == 1
        assert second.sequence_id == 2

    def test_download_failure_is_wrapped_in_download_error(self, tmp_path, monkeypatch, registry):
        def failing_run_ytdlp(self, url, dest_dir, filename_stem):
            raise RuntimeError("network unreachable")

        downloader = YtDlpDownloader(registry=registry)
        monkeypatch.setattr(YtDlpDownloader, "_run_ytdlp", failing_run_ytdlp)

        with pytest.raises(DownloadError, match="network unreachable"):
            downloader.download(URL, dest_dir=tmp_path)

    @pytest.mark.parametrize(
        "raw_message",
        [
            "Failed to resolve 'totallyfakehost123.invalid' ([Errno -2] Name or service not known)",
            "<urlopen error [Errno 111] Connection refused>",
            "HTTPSConnectionPool: Read timed out.",
        ],
    )
    def test_unreachable_host_gets_a_clear_hint(self, tmp_path, monkeypatch, registry, raw_message):
        def failing_run_ytdlp(self, url, dest_dir, filename_stem):
            raise RuntimeError(raw_message)

        downloader = YtDlpDownloader(registry=registry)
        monkeypatch.setattr(YtDlpDownloader, "_run_ytdlp", failing_run_ytdlp)

        with pytest.raises(DownloadError, match="could not be reached") as exc_info:
            downloader.download(URL, dest_dir=tmp_path)
        # the original yt-dlp/urllib message is never hidden, only prefixed
        assert raw_message in str(exc_info.value)

    @pytest.mark.parametrize(
        "raw_message",
        [
            "HTTP Error 404: Not Found",
            "ERROR: [youtube] abc123: Video unavailable. This video has been removed",
            "ERROR: Unsupported URL: https://example.com/nonsense",
            # yt-dlp's actual real-world phrasing for a bogus video ID --
            # note: no literal "video unavailable" substring, just "is
            # unavailable" -- caught this via a real (unmocked) run.
            "ERROR: [youtube] 00000000000: This video is unavailable",
        ],
    )
    def test_video_not_found_gets_a_clear_hint(self, tmp_path, monkeypatch, registry, raw_message):
        def failing_run_ytdlp(self, url, dest_dir, filename_stem):
            raise RuntimeError(raw_message)

        downloader = YtDlpDownloader(registry=registry)
        monkeypatch.setattr(YtDlpDownloader, "_run_ytdlp", failing_run_ytdlp)

        with pytest.raises(DownloadError, match="no video could be found") as exc_info:
            downloader.download(URL, dest_dir=tmp_path)
        assert raw_message in str(exc_info.value)

    def test_unrecognized_failure_falls_back_to_generic_message(
        self, tmp_path, monkeypatch, registry
    ):
        def failing_run_ytdlp(self, url, dest_dir, filename_stem):
            raise RuntimeError("some totally unrelated internal error")

        downloader = YtDlpDownloader(registry=registry)
        monkeypatch.setattr(YtDlpDownloader, "_run_ytdlp", failing_run_ytdlp)

        with pytest.raises(DownloadError, match="failed to download") as exc_info:
            downloader.download(URL, dest_dir=tmp_path)
        message = str(exc_info.value)
        assert "could not be reached" not in message
        assert "no video could be found" not in message

    def test_probe_failure_is_wrapped_in_download_error(self, tmp_path, monkeypatch, registry):
        downloader = YtDlpDownloader(registry=registry)
        _stub_run_ytdlp(monkeypatch, downloader, tmp_path, [])

        def failing_probe(self, file_path):
            raise OSError("not a valid video file")

        monkeypatch.setattr(YtDlpDownloader, "_probe", failing_probe)

        with pytest.raises(DownloadError, match="not a valid video file"):
            downloader.download(URL, dest_dir=tmp_path)

    def test_registry_is_lazily_created_and_not_touched_by_bare_instantiation(self, tmp_path):
        registry_path = tmp_path / "nested" / "registry.db"
        YtDlpDownloader(registry_path=registry_path)  # constructor only
        assert not registry_path.exists()

    def test_default_registry_constructed_lazily_from_registry_path(
        self, tmp_path, monkeypatch
    ):
        registry_path = tmp_path / "registry.db"
        downloader = YtDlpDownloader(registry_path=registry_path)
        _stub_run_ytdlp(monkeypatch, downloader, tmp_path, [])
        _stub_probe(monkeypatch)

        downloader.download(URL, dest_dir=tmp_path)

        assert registry_path.exists()
