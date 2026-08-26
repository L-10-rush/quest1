"""Unit tests for render_image_ansi -- pure function, real cv2 resize/color
conversion (cheap, deterministic, no network/model), no mocking needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.terminal_image import render_image_ansi


class TestRenderImageAnsi:
    def test_returns_multiple_lines_for_a_normal_image(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        result = render_image_ansi(image, max_width=20)
        lines = result.split("\n")
        assert len(lines) > 1

    def test_each_line_ends_with_reset_code(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        result = render_image_ansi(image, max_width=10)
        for line in result.split("\n"):
            assert line.endswith("\033[0m")

    def test_output_contains_truecolor_escape_codes(self):
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = render_image_ansi(image, max_width=10)
        assert "\033[38;2;" in result  # foreground truecolor
        assert "\033[48;2;" in result  # background truecolor

    def test_respects_max_width_upper_bound(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = render_image_ansi(image, max_width=30)
        first_line = result.split("\n")[0]
        # each terminal cell is one glyph preceded by two escape sequences --
        # count glyphs, not raw character length, against max_width
        glyph_count = first_line.count("▀")
        assert glyph_count <= 30

    def test_never_upscales_a_smaller_image(self):
        # a 5px-wide image asked for at max_width=60 should stay small,
        # not be stretched out to 60 columns
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        result = render_image_ansi(image, max_width=60)
        first_line = result.split("\n")[0]
        assert first_line.count("▀") <= 5

    def test_zero_width_image_returns_empty_string(self):
        image = np.zeros((10, 0, 3), dtype=np.uint8)
        assert render_image_ansi(image, max_width=10) == ""

    def test_zero_height_image_returns_empty_string(self):
        image = np.zeros((0, 10, 3), dtype=np.uint8)
        assert render_image_ansi(image, max_width=10) == ""

    def test_single_pixel_image_does_not_crash(self):
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        result = render_image_ansi(image, max_width=10)
        assert result != ""

    @pytest.mark.parametrize("width", [1, 2, 59, 60, 200])
    def test_various_widths_do_not_crash(self, width):
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = render_image_ansi(image, max_width=width)
        assert isinstance(result, str)
