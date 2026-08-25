"""Seconds <-> HH:MM:SS.sss formatting.

Pulled out of frame_locator so it's independently testable and reusable
anywhere a human-readable timestamp is needed (logs, JSON output, CLI print).
"""

from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    """Convert seconds to a zero-padded ``HH:MM:SS.sss`` string.

    Matches the exact format required by the problem statement's example
    output. Negative input is invalid (a video position can't be negative).
    """
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")

    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, ms = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def seconds_to_frame_number(seconds: float, fps: float) -> int:
    """Map a timestamp to the nearest frame index: ``round(seconds * fps)``.

    This is the one-line formula the whole "audio timestamp -> exact frame"
    approach hinges on (see approach.md, section 3) -- kept as a named,
    tested function rather than inlined so the mapping is unambiguous and
    reused identically by both the frame locator and any diagnostics.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return max(0, round(seconds * fps))
