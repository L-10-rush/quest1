"""Persistent registry mapping video URLs to downloaded files and metadata.

Solves two things asked of the ingestion layer:

1. **Sequenced naming** -- every downloaded video gets a short, stable,
   collision-free filename stem built from a database-assigned
   AUTOINCREMENT counter (e.g. ``000042_248244667877.mp4``), instead of
   relying solely on a platform-parsed ID that may be missing, reused, or
   unsafe as a filename for an arbitrary URL.
2. **Fast repeat lookups** -- ``find_by_url`` is an indexed point query on
   a UNIQUE column (SQLite auto-indexes UNIQUE columns), so re-running the
   pipeline against a URL it has already fetched skips the network call
   and yt-dlp entirely, rather than re-deriving anything or scanning the
   filesystem.

Uses only the stdlib `sqlite3` module -- no new dependency, and SQLite's
own file locking makes concurrent CLI invocations against the same
registry safe without extra plumbing.

Caveat (documented, not silently patched): the cache is only useful across
runs if the downloaded video file survives. `pipeline.py` deletes work
files after each run unless `--keep-work-files` is passed -- so by
default, only *within* a single process (e.g. a caller invoking
`download()` twice) does the cache actually short-circuit a re-download.
Pass `--keep-work-files` (or point `--work-dir` at a persistent location)
to get cross-run caching.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.base import VideoMetadata

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    sequence_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash         TEXT NOT NULL UNIQUE,
    source_url       TEXT NOT NULL,
    video_id         TEXT NOT NULL,
    file_path        TEXT,
    title            TEXT,
    duration_seconds REAL,
    fps              REAL,
    width            INTEGER,
    height           INTEGER,
    created_at       TEXT NOT NULL
);
"""


def _hash_url(url: str) -> str:
    """Normalize + hash a URL into the lookup key.

    Hashed (rather than storing the raw URL as the unique key) so lookups
    are a fixed-size indexed comparison regardless of URL length, and so
    trivial formatting differences don't matter once normalized.
    """
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


class VideoRegistry:
    """SQLite-backed store of every video this pipeline has downloaded."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def find_by_url(self, url: str) -> VideoMetadata | None:
        """Return cached metadata for `url` if it was already downloaded
        AND its file is still present on disk; otherwise None.

        Self-healing: a row whose file has since been deleted is treated
        as a cache miss and removed here, so the caller falls back to a
        fresh download instead of returning a dangling file path.
        """
        url_hash = _hash_url(url)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE url_hash = ?", (url_hash,)
            ).fetchone()
            if row is None or row["file_path"] is None:
                return None

            file_path = Path(row["file_path"])
            if not file_path.exists():
                logger.debug(
                    "registry entry for %s points at a missing file (%s) -- "
                    "treating as a cache miss and clearing the stale row",
                    url,
                    file_path,
                )
                conn.execute(
                    "DELETE FROM videos WHERE sequence_id = ?", (row["sequence_id"],)
                )
                conn.commit()
                return None

            return VideoMetadata(
                video_id=row["video_id"],
                source_url=row["source_url"],
                file_path=file_path,
                title=row["title"],
                duration_seconds=row["duration_seconds"],
                fps=row["fps"],
                width=row["width"],
                height=row["height"],
                sequence_id=row["sequence_id"],
            )

    def reserve(self, url: str, video_id: str) -> int:
        """Allocate (or reuse) a sequence_id for `url`, before downloading.

        Called before the actual download so the assigned number can be
        baked into the destination filename. Idempotent: calling it twice
        for the same URL returns the same sequence_id rather than burning
        a new one, via an upsert on the unique `url_hash` column.
        """
        url_hash = _hash_url(url)
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO videos (url_hash, source_url, video_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET video_id = excluded.video_id
                """,
                (url_hash, url, video_id, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT sequence_id FROM videos WHERE url_hash = ?", (url_hash,)
            ).fetchone()
            return row["sequence_id"]

    def finalize(self, url: str, metadata: VideoMetadata) -> None:
        """Write full metadata into the row `reserve()` allocated, once the
        download has actually completed."""
        url_hash = _hash_url(url)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE videos
                SET file_path = ?, title = ?, duration_seconds = ?, fps = ?,
                    width = ?, height = ?
                WHERE url_hash = ?
                """,
                (
                    str(metadata.file_path),
                    metadata.title,
                    metadata.duration_seconds,
                    metadata.fps,
                    metadata.width,
                    metadata.height,
                    url_hash,
                ),
            )
            conn.commit()

    @staticmethod
    def filename_stem(sequence_id: int, video_id: str) -> str:
        """Build the sequenced filename stem, e.g. ``000042_248244667877``."""
        return f"{sequence_id:06d}_{video_id}"
