"""Unit tests for the CLI entrypoint / composition root (src/main.py).

Sprint 3 (robustness): proves the CLI never crashes with a raw traceback --
a `PipelineError` maps to exit code 1, an unimplemented scaffold stage maps
to exit code 2, and a successful run prints the exact fields the problem
statement's example output requires. `build_pipeline` itself is mocked out
so these tests never touch the network or a real model.

Also covers the interactive session loop added on top of `--url`/`--text`:
when `--text` is omitted, `main()` preprocesses the video once
(`prepare()`) and then repeatedly calls `locate_dialogue()` for each line
typed at the `dialogue>` prompt until the user exits.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.main as main_module
from src.config import PipelineConfig
from src.exceptions import DownloadError, ResultPersistenceError
from src.pipeline import PipelineRunSummary, PreparedSession
from src.transcription.vosk_engine import VoskEngine
from src.transcription.whisperx_engine import WhisperXEngine

MIN_ARGS = ["--url", "https://ok.ru/video/248244667877", "--text", "My mind rebels at stagnation"]
URL_ONLY_ARGS = ["--url", "https://ok.ru/video/248244667877"]

FAKE_SESSION = PreparedSession(
    video=SimpleNamespace(title="Test Video"), audio=None, transcript=None, metrics=None
)


class _FakePipeline:
    """Stands in for `DialoguePipeline`: `prepare_outcome` is returned by
    (or raised from) `prepare()`; `locate_outcomes` is consumed in order,
    one per `locate_dialogue()` call -- one entry per dialogue line the
    test expects to be searched."""

    def __init__(self, prepare_outcome=FAKE_SESSION, locate_outcomes=None):
        self._prepare_outcome = prepare_outcome
        self._locate_outcomes = list(locate_outcomes or [])
        self.locate_calls: list[tuple] = []
        self.cleanup_calls: list = []

    def prepare(self):
        if isinstance(self._prepare_outcome, Exception):
            raise self._prepare_outcome
        return self._prepare_outcome

    def locate_dialogue(self, session, target_text):
        self.locate_calls.append((session, target_text))
        outcome = self._locate_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def cleanup(self, session):
        self.cleanup_calls.append(session)


def _stub_build_pipeline(monkeypatch, prepare_outcome=FAKE_SESSION, locate_outcomes=None):
    fake = _FakePipeline(prepare_outcome, locate_outcomes)
    monkeypatch.setattr(main_module, "build_pipeline", lambda config: fake)
    return fake


SUCCESS_SUMMARY = PipelineRunSummary(
    timestamp="00:00:42.360",
    frame_number=1059,
    matched_text="My mind rebels at stagnation",
    match_score=96.5,
    is_uncertain=False,
    uncertainty_reason=None,
    result_json_path="output/248244667877/results/result_1059.json",
    frame_image_path="output/248244667877/frames/frame_1059.png",
    total_seconds=12.3,
)


class TestBuildTranscriber:
    def test_whisperx_engine_selected_by_default(self):
        config = PipelineConfig(video_url="https://example.com/v", target_text="hi")
        assert isinstance(main_module._build_transcriber(config), WhisperXEngine)

    def test_vosk_engine_selected_when_configured(self):
        config = PipelineConfig(video_url="https://example.com/v", target_text="hi", engine="vosk")
        assert isinstance(main_module._build_transcriber(config), VoskEngine)


class TestMainExitCodes:
    """`--url` + `--text` given: the previous single-shot behavior."""

    def test_returns_0_and_prints_summary_on_success(self, monkeypatch, capsys):
        fake = _stub_build_pipeline(monkeypatch, locate_outcomes=[SUCCESS_SUMMARY])

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Timestamp : 00:00:42.360" in out
        assert "Frame     : 1059" in out
        assert 'Text      : "My mind rebels at stagnation"' in out
        assert "output/248244667877/frames/frame_1059.png" in out
        assert fake.locate_calls == [(FAKE_SESSION, "My mind rebels at stagnation")]
        assert fake.cleanup_calls == [FAKE_SESSION]

    def test_uncertain_result_prints_a_warning_line(self, monkeypatch, capsys):
        uncertain_summary = PipelineRunSummary(
            timestamp="00:00:01.000",
            frame_number=25,
            matched_text="something else",
            match_score=42.0,
            is_uncertain=True,
            uncertainty_reason="best match scored 42.0, below threshold 80.0",
            result_json_path="output/x/results/result_25.json",
            frame_image_path="output/x/frames/frame_25.png",
            total_seconds=5.0,
        )
        _stub_build_pipeline(monkeypatch, locate_outcomes=[uncertain_summary])

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 0  # uncertain is still a successful run, just flagged
        out = capsys.readouterr().out
        assert "UNCERTAIN" in out
        assert "below threshold" in out

    def test_returns_1_when_prepare_fails(self, monkeypatch, capsys):
        # NOTE: main() calls configure_logging(), which installs its own
        # stderr handler and clears root.handlers -- that also strips
        # pytest's `caplog` handler, so we assert on real captured stderr
        # (capsys) rather than caplog here.
        fake = _stub_build_pipeline(monkeypatch, prepare_outcome=DownloadError("network unreachable"))

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 1
        assert "network unreachable" in capsys.readouterr().err
        # never got a session, so nothing to clean up
        assert fake.cleanup_calls == []

    def test_returns_1_when_search_fails(self, monkeypatch, capsys):
        fake = _stub_build_pipeline(
            monkeypatch, locate_outcomes=[ResultPersistenceError("disk full")]
        )

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 1
        assert "disk full" in capsys.readouterr().err
        # session WAS obtained, so cleanup still runs (via the finally block)
        assert fake.cleanup_calls == [FAKE_SESSION]

    def test_returns_2_on_not_implemented_scaffold_stage(self, monkeypatch, capsys):
        _stub_build_pipeline(
            monkeypatch, prepare_outcome=NotImplementedError("FfmpegAudioExtractor is a scaffold")
        )

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 2
        assert "scaffold" in capsys.readouterr().err

    def test_missing_url_exits_before_reaching_the_pipeline(self, monkeypatch):
        called = False

        def fail_if_called(config):
            nonlocal called
            called = True

        monkeypatch.setattr(main_module, "build_pipeline", fail_if_called)

        with pytest.raises(SystemExit):
            main_module.main(["--text", "hello"])  # missing --url

        assert not called


class TestMainInteractiveSession:
    """`--url` given without `--text`: preprocess once, then loop over
    dialogue lines typed at the `dialogue>` prompt until the user exits."""

    def test_searches_each_line_until_exit(self, monkeypatch, capsys):
        fake = _stub_build_pipeline(monkeypatch, locate_outcomes=[SUCCESS_SUMMARY])
        answers = iter(["My mind rebels at stagnation", "exit"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        exit_code = main_module.main(URL_ONLY_ARGS)

        assert exit_code == 0
        assert fake.locate_calls == [(FAKE_SESSION, "My mind rebels at stagnation")]
        assert fake.cleanup_calls == [FAKE_SESSION]
        assert "Timestamp : 00:00:42.360" in capsys.readouterr().out

    def test_blank_lines_are_ignored_and_reprompt(self, monkeypatch):
        fake = _stub_build_pipeline(monkeypatch, locate_outcomes=[SUCCESS_SUMMARY])
        answers = iter(["", "   ", "hello there", "quit"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        exit_code = main_module.main(URL_ONLY_ARGS)

        assert exit_code == 0
        assert fake.locate_calls == [(FAKE_SESSION, "hello there")]

    def test_a_failed_search_does_not_end_the_session(self, monkeypatch, capsys):
        fake = _stub_build_pipeline(
            monkeypatch,
            locate_outcomes=[ResultPersistenceError("disk full"), SUCCESS_SUMMARY],
        )
        answers = iter(["first line", "second line", "exit"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        exit_code = main_module.main(URL_ONLY_ARGS)

        assert exit_code == 0
        assert [call[1] for call in fake.locate_calls] == ["first line", "second line"]
        assert "disk full" in capsys.readouterr().err
        assert fake.cleanup_calls == [FAKE_SESSION]

    def test_eof_ends_the_session_cleanly(self, monkeypatch):
        fake = _stub_build_pipeline(monkeypatch, locate_outcomes=[])

        def raise_eof(*_a, **_k):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)

        exit_code = main_module.main(URL_ONLY_ARGS)

        assert exit_code == 0
        assert fake.locate_calls == []
        assert fake.cleanup_calls == [FAKE_SESSION]
