"""VideoDownloader interface (Dependency Inversion boundary for stage 1).

`pipeline.py` depends only on this ABC, never on a concrete downloader --
that's what lets `ytdlp_downloader.py` be swapped for a different
implementation (e.g. a direct-file-path loader for local testing) without
touching pipeline.py (Open/Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    """Everything downstream stages need to know about the fetched video.

    `sequence_id` is a locally-assigned, monotonically increasing integer
    (see `ingestion/registry.py`) used to build a short, collision-free
    filename for the downloaded video regardless of source platform --
    distinct from `video_id`, which is parsed from the URL and may be
    absent/ambiguous for platforms `utils/video_id.py` doesn't recognize.
    Defaults to 0 for metadata constructed outside the registry-backed
    downloader (e.g. in tests).
    """

    video_id: str
    source_url: str
    file_path: Path
    title: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    sequence_id: int = 0


class VideoDownloader(ABC):
    """Fetches a video from a URL and reports its metadata."""

    @abstractmethod
    def download(self, url: str, dest_dir: Path) -> VideoMetadata:
        """Download `url` into `dest_dir` and return its metadata.

        Implementations MUST raise `exceptions.DownloadError` (never let a
        library-specific exception escape) so `pipeline.py` can handle every
        downloader uniformly.
        """
        raise NotImplementedError
