"""Smoke tests for the Streamlit UI (src/webapp/app.py).

Uses Streamlit's own `AppTest` harness to actually execute the script and
simulate real widget interaction (typing a URL, clicking buttons), with
`build_pipeline` mocked out the same way `test_main.py` mocks it for the
CLI -- these tests never touch the network or a real model. They only
prove the UI wiring itself (form -> pipeline calls -> rendered result)
doesn't crash and shows what a search actually produced.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Streamlit is an optional dependency (`uv sync --group web`, see
# pyproject.toml) -- skip this whole file rather than error out when only
# the base/dev groups are installed (the default `uv sync` from the README).
st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

import src.main as main_module
from src.pipeline import PipelineRunSummary, PreparedSession

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "webapp" / "app.py")
TEST_URL = "https://ok.ru/video/248244667877"

FAKE_SESSION = PreparedSession(
    video=SimpleNamespace(
        title="Test Video", video_id="248244667877", duration_seconds=90.0, fps=25.0
    ),
    audio=None,
    transcript=None,
    metrics=None,
)


class _FakePipeline:
    """Stands in for `DialoguePipeline`, mirroring test_main.py's fake."""

    def __init__(self, locate_outcomes):
        self._locate_outcomes = list(locate_outcomes)
        self.cleanup_calls: list = []

    def prepare(self):
        return FAKE_SESSION

    def locate_dialogue(self, session, target_text):
        outcome = self._locate_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def cleanup(self, session):
        self.cleanup_calls.append(session)


def _summary(tmp_path, **overrides) -> PipelineRunSummary:
    """A real summary pointing at real (throwaway) files, so the download
    buttons and image rendering in `_render_result` actually get exercised
    instead of silently no-op'ing on a path that doesn't exist."""
    video_dir = tmp_path / "output" / "248244667877"
    (video_dir / "frames").mkdir(parents=True, exist_ok=True)
    (video_dir / "results").mkdir(parents=True, exist_ok=True)
    frame_path = video_dir / "frames" / "frame_1059.png"
    json_path = video_dir / "results" / "result_1059.json"
    frame_path.write_bytes(b"not a real png, just bytes for the download button")
    json_path.write_text('{"stub": true}')

    defaults = {
        "timestamp": "00:00:42.360",
        "frame_number": 1059,
        "matched_text": "My mind rebels at stagnation",
        "match_score": 96.5,
        "is_uncertain": False,
        "uncertainty_reason": None,
        "result_json_path": str(json_path),
        "frame_image_path": str(frame_path),
        "total_seconds": 1.2,
        "screen_status": "on_screen",
        "screen_confidence": 0.9,
        "screen_reason": "a face was visible with mouth movement consistent with speech",
        "best_frame_image": np.zeros((10, 10, 3), dtype=np.uint8),
    }
    defaults.update(overrides)
    return PipelineRunSummary(**defaults)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Mirrors test_main.py's fixture: main.append_session_log (reused by
    the webapp) writes a real file, so keep every test out of the repo."""
    monkeypatch.chdir(tmp_path)


def _stub_build_pipeline(monkeypatch, locate_outcomes):
    fake = _FakePipeline(locate_outcomes)
    monkeypatch.setattr(main_module, "build_pipeline", lambda config: fake)
    return fake


def _load_video(at: AppTest) -> None:
    at.text_input[0].input(TEST_URL)
    at.button[0].click()
    at.run()


def _search(at: AppTest, text: str) -> None:
    dialogue_input = next(t for t in at.text_input if t.label == "Dialogue line")
    dialogue_input.input(text)
    search_button = next(b for b in at.button if b.label == "Search")
    search_button.click()
    at.run()


class TestInitialState:
    def test_renders_without_exception_before_any_video_is_loaded(self):
        at = AppTest.from_file(APP_PATH)
        at.run()
        assert not at.exception
        assert any("video-dialogue-finder" in t.value for t in at.title)
        assert not any(f.label == "Dialogue line" for f in at.text_input)


class TestLoadVideo:
    def test_successful_prepare_shows_the_video_title_and_search_form(self, monkeypatch):
        _stub_build_pipeline(monkeypatch, locate_outcomes=[])
        at = AppTest.from_file(APP_PATH)
        at.run()

        _load_video(at)

        assert not at.exception
        assert any("Test Video" in s.value for s in at.success)
        assert any(t.label == "Dialogue line" for t in at.text_input)

    def test_prepare_failure_shows_an_error_not_a_traceback(self, monkeypatch):
        from src.exceptions import DownloadError

        class _FailingPipeline:
            def prepare(self):
                raise DownloadError("network unreachable")

        monkeypatch.setattr(main_module, "build_pipeline", lambda config: _FailingPipeline())

        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].input(TEST_URL)
        at.button[0].click()
        at.run()

        assert not at.exception
        assert any("network unreachable" in e.value for e in at.error)
        assert not any(t.label == "Dialogue line" for t in at.text_input)

    def test_malformed_url_shows_an_error_not_a_crash(self, monkeypatch):
        """PipelineConfig's own URL-format validation (ValueError) must be
        caught same as a PipelineError -- a bad URL typed into the sidebar
        should render `st.error`, never an unhandled-exception page."""
        called = False

        def fail_if_called(config):
            nonlocal called
            called = True

        monkeypatch.setattr(main_module, "build_pipeline", fail_if_called)

        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].input("not-a-url")
        at.button[0].click()
        at.run()

        assert not at.exception
        assert not called
        assert any("valid http" in e.value for e in at.error)
        assert not any(t.label == "Dialogue line" for t in at.text_input)


class TestSearch:
    def test_search_renders_the_matched_frame_and_metrics(self, monkeypatch, tmp_path):
        summary = _summary(tmp_path)
        _stub_build_pipeline(monkeypatch, locate_outcomes=[summary])

        at = AppTest.from_file(APP_PATH)
        at.run()
        _load_video(at)
        _search(at, "My mind rebels at stagnation")

        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Timestamp"] == "00:00:42.360"
        assert metrics["Frame"] == "1059"
        assert metrics["Match score"] == "96.5"
        assert len(at.image) == 1
        assert any("session.log" in c.value for c in at.caption)

    def test_uncertain_match_is_flagged_not_hidden(self, monkeypatch, tmp_path):
        summary = _summary(
            tmp_path, is_uncertain=True, uncertainty_reason="low confidence score"
        )
        _stub_build_pipeline(monkeypatch, locate_outcomes=[summary])

        at = AppTest.from_file(APP_PATH)
        at.run()
        _load_video(at)
        _search(at, "an ambiguous line")

        assert not at.exception
        assert any("UNCERTAIN" in w.value and "low confidence score" in w.value for w in at.warning)

    def test_off_screen_verdict_shown_as_error_badge(self, monkeypatch, tmp_path):
        summary = _summary(
            tmp_path,
            screen_status="off_screen",
            screen_confidence=1.0,
            screen_reason="no face detected in any sampled frame",
        )
        _stub_build_pipeline(monkeypatch, locate_outcomes=[summary])

        at = AppTest.from_file(APP_PATH)
        at.run()
        _load_video(at)
        _search(at, "a narrated line")

        assert not at.exception
        assert any("OFF_SCREEN" in e.value for e in at.error)

    def test_search_failure_shows_an_error_and_keeps_the_session_alive(self, monkeypatch, tmp_path):
        from src.exceptions import MatchingError

        _stub_build_pipeline(monkeypatch, locate_outcomes=[MatchingError("matcher exploded")])

        at = AppTest.from_file(APP_PATH)
        at.run()
        _load_video(at)
        _search(at, "anything")

        assert not at.exception
        assert any("matcher exploded" in e.value for e in at.error)
        # The session (and its "Dialogue line" search form) survives a
        # failed search -- same contract as the CLI's interactive loop.
        assert any(t.label == "Dialogue line" for t in at.text_input)

    def test_candidate_previews_render_when_ambiguous(self, monkeypatch, tmp_path):
        candidates = (
            ("my mind rebels, at stagnation", 91.0, np.zeros((10, 10, 3), dtype=np.uint8)),
            ("my mind rebels at stagnation now", 85.0, np.zeros((10, 10, 3), dtype=np.uint8)),
        )
        summary = _summary(tmp_path, candidate_previews=candidates)
        _stub_build_pipeline(monkeypatch, locate_outcomes=[summary])

        at = AppTest.from_file(APP_PATH)
        at.run()
        _load_video(at)
        _search(at, "My mind rebels at stagnation")

        assert not at.exception
        # 1 best-match image + 2 candidate previews.
        assert len(at.image) == 3
        assert any("Other candidates (2" in exp.label for exp in at.expander)


class TestReset:
    def test_reset_button_cleans_up_and_restores_the_load_form(self, monkeypatch):
        fake = _stub_build_pipeline(monkeypatch, locate_outcomes=[])
        at = AppTest.from_file(APP_PATH)
        at.run()
        _load_video(at)

        reset_button = next(b for b in at.button if "New video" in b.label)
        reset_button.click()
        at.run()

        assert not at.exception
        assert fake.cleanup_calls == [FAKE_SESSION]
        assert not any(t.label == "Dialogue line" for t in at.text_input)
        assert at.text_input[0].disabled is False
