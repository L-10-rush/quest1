"""Derive a stable, filesystem-safe ID for a video URL.

Used to key `output/<video_id>/` so that re-running the pipeline against the
same URL with a different target phrase reuses the same output folder, and
different URLs never collide.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlparse

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def extract_video_id(url: str) -> str:
    """Return a short, stable, filesystem-safe identifier for `url`.

    Tries known platform URL shapes first (readable IDs like ``dQw4w9WgXcQ``
    or ``248244667877``); falls back to a 12-char hash of the URL so *any*
    URL still gets a stable, collision-resistant ID (required since the
    evaluators may substitute a different video/platform entirely).
    """
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")

    if host in _YOUTUBE_HOSTS or host == "youtu.be":
        if host == "youtu.be":
            candidate = parsed.path.strip("/")
            if candidate:
                return _sanitize(candidate)
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return _sanitize(query_id)

    if "ok.ru" in host:
        match = re.search(r"/video/(\d+)", parsed.path)
        if match:
            return match.group(1)

    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:12]


def _sanitize(candidate: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", candidate)
