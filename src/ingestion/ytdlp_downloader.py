"""yt-dlp-backed VideoDownloader, cached and sequence-named via VideoRegistry.

Works for any site yt-dlp supports (YouTube, ok.ru, ...) without
per-platform branching -- yt-dlp itself handles extraction. Every download
is registered in a `VideoRegistry` (SQLite) so a repeat request for the
same URL returns the cached `VideoMetadata` directly, and every downloaded
file is named with a database-assigned sequence number rather than a
raw/possibly-unsafe platform ID.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import yt_dlp

from src.exceptions import DownloadError
from src.ingestion.base import VideoDownloader, VideoMetadata
from src.ingestion.registry import VideoRegistry
from src.utils.video_id import extract_video_id

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("output/registry.db")


class YtDlpDownloader(VideoDownloader):
    """Downloads video via `yt-dlp` and probes the result with OpenCV.

    The registry is lazily constructed on first use (not in `__init__`)
    so simply instantiating this class -- e.g. in a test that exercises
    something else -- never touches disk.
    """

    def __init__(
        self,
        # Adapted from yt-dlp's own default (`bv*+ba/b`, used when no `-f`
        # is given at all) with one addition: prefer an h264/avc1 video
        # stream over AV1 when both exist. A stricter selector like
        # `best[ext=mp4]/best` requiring a single progressive stream
        # fails outright on most modern YouTube videos, which serve
        # video and audio as separate DASH streams and often have no
        # muxed "best" format at all -- see the `merge_output_format`
        # note in `_run_ytdlp` for why merging separate streams is safe
        # here. The avc1 preference exists because OpenCV's bundled
        # ffmpeg build has an unreliable AV1 decoder/seeker (observed:
        # garbage `CAP_PROP_POS_FRAMES` reads and "Missing Sequence
        # Header" errors on an AV1-only download) -- h264 has by far the
        # widest, most reliable decode+seek support across ffmpeg/OpenCV
        # builds, and YouTube serves it for nearly every video.
        format_selector: str = "bestvideo*[vcodec^=avc1]+bestaudio/bestvideo*+bestaudio/best",
        registry: VideoRegistry | None = None,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
    ) -> None:
        self._format_selector = format_selector
        self._registry = registry
        self._registry_path = registry_path

    def _get_registry(self) -> VideoRegistry:
        if self._registry is None:
            self._registry = VideoRegistry(self._registry_path)
        return self._registry

    def download(self, url: str, dest_dir: Path) -> VideoMetadata:
        registry = self._get_registry()

        cached = registry.find_by_url(url)
        if cached is not None:
            logger.info(
                "cache hit for %s -> %s (sequence #%06d)",
                url,
                cached.file_path,
                cached.sequence_id,
            )
            return cached

        dest_dir.mkdir(parents=True, exist_ok=True)
        video_id = extract_video_id(url)
        sequence_id = registry.reserve(url, video_id)
        filename_stem = registry.filename_stem(sequence_id, video_id)

        try:
            file_path, title = self._run_ytdlp(url, dest_dir, filename_stem)
            fps, duration, width, height = self._probe(file_path)
        except Exception as exc:
            # yt-dlp raises many distinct exception types depending on the
            # failure (network, extractor, unsupported URL, ...) and OpenCV
            # raises its own on a corrupt/empty output file -- deliberately
            # broad here so every one of them is normalized to DownloadError
            # (see the contract in ingestion/base.py: callers only ever
            # need to handle one exception type).
            raise DownloadError(f"failed to download {url}: {exc}") from exc

        metadata = VideoMetadata(
            video_id=video_id,
            source_url=url,
            file_path=file_path,
            title=title,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            sequence_id=sequence_id,
        )
        registry.finalize(url, metadata)
        logger.info("downloaded %s -> %s (sequence #%06d)", url, file_path, sequence_id)
        return metadata

    def _run_ytdlp(self, url: str, dest_dir: Path, filename_stem: str) -> tuple[Path, str]:
        opts = {
            "format": self._format_selector,
            "outtmpl": str(dest_dir / f"{filename_stem}.%(ext)s"),
            # When the format selector picks separate video/audio streams,
            # yt-dlp shells out to `ffmpeg` (already a hard dependency,
            # see README) to mux them -- forcing the container to mp4
            # keeps the downloaded file's extension predictable regardless
            # of source codecs (a remux, e.g. VP9+Opus into mp4, is not
            # strictly to-spec mp4 but both ffmpeg and OpenCV -- our only
            # downstream readers -- handle it fine).
            "merge_output_format": "mp4",
            "quiet": not logger.isEnabledFor(logging.DEBUG),
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = Path(ydl.prepare_filename(info))
            title = info.get("title") or filename_stem
        return file_path, title

    def _probe(self, file_path: Path) -> tuple[float, float, int, int]:
        """Probe the *downloaded file* for ground-truth fps/duration/size
        rather than trusting yt-dlp's reported metadata, which can differ
        from the actual re-encoded/muxed output."""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise OSError(f"downloaded file could not be opened by OpenCV: {file_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return fps, duration, width, height
