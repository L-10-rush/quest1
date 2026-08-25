"""Unit tests for VideoRegistry: sequenced IDs, indexed URL lookup, and
self-healing behaviour when a cached file has been deleted.

Pure sqlite3 + tmp_path -- no mocking needed, no network involved.
"""

from pathlib import Path

from src.ingestion.base import VideoMetadata
from src.ingestion.registry import VideoRegistry, _hash_url


def _metadata(video_id: str, file_path: Path, sequence_id: int = 0) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        source_url=f"https://example.com/{video_id}",
        file_path=file_path,
        title=f"title-{video_id}",
        duration_seconds=12.5,
        fps=25.0,
        width=1280,
        height=720,
        sequence_id=sequence_id,
    )


class TestReserve:
    def test_first_reservation_starts_at_one(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        assert registry.reserve("https://example.com/a", "a") == 1

    def test_sequential_across_different_urls(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        first = registry.reserve("https://example.com/a", "a")
        second = registry.reserve("https://example.com/b", "b")
        third = registry.reserve("https://example.com/c", "c")
        assert (first, second, third) == (1, 2, 3)

    def test_idempotent_for_the_same_url(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        first = registry.reserve("https://example.com/a", "a")
        second = registry.reserve("https://example.com/a", "a")
        assert first == second


class TestFilenameStem:
    def test_zero_padded_six_digits(self):
        assert VideoRegistry.filename_stem(1, "248244667877") == "000001_248244667877"

    def test_large_sequence_id_not_truncated(self):
        assert VideoRegistry.filename_stem(1_234_567, "x") == "1234567_x"


class TestFindByUrl:
    def test_returns_none_for_unknown_url(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        assert registry.find_by_url("https://example.com/never-seen") is None

    def test_returns_none_after_reserve_but_before_finalize(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        registry.reserve("https://example.com/a", "a")
        assert registry.find_by_url("https://example.com/a") is None

    def test_returns_cached_metadata_after_finalize(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        url = "https://example.com/a"
        seq = registry.reserve(url, "a")
        video_file = tmp_path / "a.mp4"
        video_file.write_bytes(b"fake video bytes")

        registry.finalize(url, _metadata("a", video_file, sequence_id=seq))

        cached = registry.find_by_url(url)
        assert cached is not None
        assert cached.video_id == "a"
        assert cached.file_path == video_file
        assert cached.sequence_id == seq
        assert cached.duration_seconds == 12.5

    def test_self_heals_when_cached_file_is_missing(self, tmp_path: Path):
        registry = VideoRegistry(tmp_path / "registry.db")
        url = "https://example.com/a"
        seq = registry.reserve(url, "a")
        video_file = tmp_path / "a.mp4"
        video_file.write_bytes(b"fake video bytes")
        registry.finalize(url, _metadata("a", video_file, sequence_id=seq))

        video_file.unlink()  # simulate work-file cleanup between runs

        assert registry.find_by_url(url) is None

    def test_stale_row_is_removed_so_a_fresh_reserve_gets_a_new_sequence_id(
        self, tmp_path: Path
    ):
        registry = VideoRegistry(tmp_path / "registry.db")
        url = "https://example.com/a"
        first_seq = registry.reserve(url, "a")
        video_file = tmp_path / "a.mp4"
        video_file.write_bytes(b"x")
        registry.finalize(url, _metadata("a", video_file, sequence_id=first_seq))
        video_file.unlink()

        assert registry.find_by_url(url) is None  # triggers self-heal deletion
        second_seq = registry.reserve(url, "a")
        assert second_seq != first_seq


class TestPersistence:
    def test_survives_across_separate_registry_instances(self, tmp_path: Path):
        db_path = tmp_path / "registry.db"
        url = "https://example.com/a"
        video_file = tmp_path / "a.mp4"
        video_file.write_bytes(b"x")

        writer = VideoRegistry(db_path)
        seq = writer.reserve(url, "a")
        writer.finalize(url, _metadata("a", video_file, sequence_id=seq))

        reader = VideoRegistry(db_path)  # fresh instance, same file on disk
        cached = reader.find_by_url(url)
        assert cached is not None
        assert cached.sequence_id == seq


class TestHashUrl:
    def test_deterministic(self):
        assert _hash_url("https://example.com/a") == _hash_url("https://example.com/a")

    def test_different_urls_hash_differently(self):
        assert _hash_url("https://example.com/a") != _hash_url("https://example.com/b")

    def test_whitespace_normalized(self):
        assert _hash_url("https://example.com/a") == _hash_url("  https://example.com/a  ")
