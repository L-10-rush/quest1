from src.utils.video_id import extract_video_id


class TestExtractVideoId:
    def test_okru_url(self):
        assert extract_video_id("https://ok.ru/video/248244667877") == "248244667877"

    def test_youtube_watch_url(self):
        assert extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"
        ) == "dQw4w9WgXcQ"

    def test_youtube_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_unknown_platform_falls_back_to_stable_hash(self):
        url = "https://example.com/some/weird/video/path"
        id_1 = extract_video_id(url)
        id_2 = extract_video_id(url)
        assert id_1 == id_2
        assert len(id_1) == 12

    def test_different_urls_get_different_fallback_ids(self):
        assert extract_video_id("https://example.com/a") != extract_video_id(
            "https://example.com/b"
        )

    def test_fallback_id_is_filesystem_safe(self):
        video_id = extract_video_id("https://example.com/a?b=c&d=e")
        assert all(c.isalnum() for c in video_id)
