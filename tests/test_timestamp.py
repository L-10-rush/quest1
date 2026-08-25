import pytest

from src.utils.timestamp import format_timestamp, seconds_to_frame_number


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00.000"

    def test_sub_second(self):
        assert format_timestamp(0.5) == "00:00:00.500"

    def test_hours_minutes_seconds_ms(self):
        # 1h 1m 1.5s
        assert format_timestamp(3661.5) == "01:01:01.500"

    def test_rounds_to_nearest_millisecond(self):
        assert format_timestamp(1.0004) == "00:00:01.000"
        assert format_timestamp(1.0006) == "00:00:01.001"

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            format_timestamp(-1)


class TestSecondsToFrameNumber:
    def test_basic(self):
        assert seconds_to_frame_number(1.0, 25.0) == 25

    def test_rounds_to_nearest_frame(self):
        # 1.501s * 24fps = 36.024 -> rounds to 36
        assert seconds_to_frame_number(1.501, 24.0) == 36

    def test_zero_seconds(self):
        assert seconds_to_frame_number(0.0, 30.0) == 0

    def test_invalid_fps_rejected(self):
        with pytest.raises(ValueError):
            seconds_to_frame_number(1.0, 0)
