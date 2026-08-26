"""Render a small preview of a frame directly in the terminal, using
24-bit-color half-block characters -- no image-protocol support required
(works over plain SSH too), and no dependency beyond `cv2`/`numpy`, which
the pipeline already needs for frame extraction.

Not a substitute for opening the saved PNG (`Image` in the CLI summary) --
just enough of a preview, inline, that "was this the right frame" and
"which of these candidates looks right" don't require leaving the terminal.
"""

from __future__ import annotations

import cv2
import numpy as np

_HALF_BLOCK = "▀"  # upper half block: foreground paints the top pixel, background the bottom
_RESET = "\033[0m"

#: Terminal character cells are roughly twice as tall as they are wide, so
#: this shrinks the vertical scale again on top of the half-block trick's
#: own 2x vertical doubling, keeping the preview from looking stretched.
_CELL_ASPECT_CORRECTION = 0.5


def render_image_ansi(image_bgr: np.ndarray, max_width: int = 60) -> str:
    """Return a multi-line ANSI-colored string previewing `image_bgr` (as
    returned by OpenCV, i.e. BGR channel order) at up to `max_width`
    terminal columns wide.

    Each output row encodes two source pixel rows -- one as the half-block
    glyph's foreground color, one as its background -- so vertical
    resolution is effectively doubled for the same number of terminal
    lines. Returns an empty string for a degenerate (zero-sized) image
    rather than raising, since a preview is best-effort by nature.
    """
    height, width = image_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return ""

    scale = min(1.0, max_width / width)
    out_width = max(1, round(width * scale))
    out_height = max(2, round(height * scale * _CELL_ASPECT_CORRECTION))
    if out_height % 2:
        out_height += 1  # need an even number of source rows to pair into half-blocks

    resized = cv2.resize(image_bgr, (out_width, out_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    lines = []
    for y in range(0, out_height, 2):
        top, bottom = rgb[y], rgb[y + 1]
        cells = [
            f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m{_HALF_BLOCK}"
            for (tr, tg, tb), (br, bg, bb) in zip(top, bottom)
        ]
        lines.append("".join(cells) + _RESET)
    return "\n".join(lines)
