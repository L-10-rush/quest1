"""yt-dlp-backed VideoDownloader.

STATUS: scaffold only -- `download()` is intentionally left unimplemented.
Wiring, typing, and error contract are done; the actual yt-dlp invocation
and metadata probing require a live network call against the real
assignment URL to validate, so that judgment call is left for you rather
than guessed at here. See the TODO in `download()` for the exact steps.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.exceptions import DownloadError
from src.ingestion.base import VideoDownloader, VideoMetadata
from src.utils.video_id import extract_video_id

logger = logging.getLogger(__name__)


class YtDlpDownloader(VideoDownloader):
    """Downloads video via the `yt-dlp` library and probes it with OpenCV.

    Works for any site yt-dlp supports (YouTube, ok.ru, ...) without
    per-platform branching -- yt-dlp itself handles extraction.
    """

    def __init__(self, format_selector: str = "best[ext=mp4]/best") -> None:
        self._format_selector = format_selector

    def download(self, url: str, dest_dir: Path) -> VideoMetadata:
        """Download `url` into `dest_dir`, return its `VideoMetadata`.

        TODO (not implemented -- fill this in and validate against the real
        assignment URL, https://ok.ru/video/248244667877):

        1. `dest_dir.mkdir(parents=True, exist_ok=True)`
        2. video_id = extract_video_id(url)  (already available, see import)
        3. Run yt-dlp, e.g.:
               import yt_dlp
               opts = {
                   "format": self._format_selector,
                   "outtmpl": str(dest_dir / f"{video_id}.%(ext)s"),
                   "quiet": not logger.isEnabledFor(logging.DEBUG),
               }
               with yt_dlp.YoutubeDL(opts) as ydl:
                   info = ydl.extract_info(url, download=True)
                   file_path = Path(ydl.prepare_filename(info))
        4. Probe the *downloaded file* for ground-truth fps/duration/size
           with OpenCV (don't trust yt-dlp's reported metadata -- it can be
           wrong for the actual re-encoded/muxed output):
               import cv2
               cap = cv2.VideoCapture(str(file_path))
               fps = cap.get(cv2.CAP_PROP_FPS)
               frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
               duration = frame_count / fps if fps else 0.0
               width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
               height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
               cap.release()
        5. Wrap steps 3-4 in try/except and raise
           `DownloadError(str(exc)) from exc` on any failure -- never let a
           yt_dlp.utils.DownloadError or cv2 error escape this method.
        6. Return `VideoMetadata(video_id=video_id, source_url=url,
           file_path=file_path, title=info.get("title", video_id),
           duration_seconds=duration, fps=fps, width=width, height=height)`
        """
        raise NotImplementedError(
            "YtDlpDownloader.download() is a scaffold -- implement steps 1-6 "
            "in the docstring above and validate against a real video URL."
        )
