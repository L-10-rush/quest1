"""Unit tests for the CLI entrypoint / composition root (src/main.py).

Sprint 3 (robustness): proves the CLI never crashes with a raw traceback --
a `PipelineError` maps to exit code 1, an unimplemented scaffold stage maps
to exit code 2, and a successful run prints the exact fields the problem
statement's example output requires. `build_pipeline` itself is mocked out
so these tests never touch the network or a real model.
"""

from __future__ import annotations

import pytest

import src.main as main_module
from src.config import PipelineConfig
from src.exceptions import DownloadError
from src.pipeline import PipelineRunSummary
from src.transcription.vosk_engine import VoskEngine
from src.transcription.whisperx_engine import WhisperXEngine

MIN_ARGS = ["--url", "https://ok.ru/video/248244667877", "--text", "My mind rebels at stagnation"]


class _FakePipeline:
    def __init__(self, outcome):
        self._outcome = outcome

    def run(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _stub_build_pipeline(monkeypatch, outcome):
    monkeypatch.setattr(main_module, "build_pipeline", lambda config: _FakePipeline(outcome))


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
    def test_returns_0_and_prints_summary_on_success(self, monkeypatch, capsys):
        _stub_build_pipeline(monkeypatch, SUCCESS_SUMMARY)

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Timestamp : 00:00:42.360" in out
        assert "Frame     : 1059" in out
        assert 'Text      : "My mind rebels at stagnation"' in out
        assert "output/248244667877/frames/frame_1059.png" in out

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
        _stub_build_pipeline(monkeypatch, uncertain_summary)

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 0  # uncertain is still a successful run, just flagged
        out = capsys.readouterr().out
        assert "UNCERTAIN" in out
        assert "below threshold" in out

    def test_returns_1_on_pipeline_error(self, monkeypatch, capsys):
        # NOTE: main() calls configure_logging(), which installs its own
        # stderr handler and clears root.handlers -- that also strips
        # pytest's `caplog` handler, so we assert on real captured stderr
        # (capsys) rather than caplog here.
        _stub_build_pipeline(monkeypatch, DownloadError("network unreachable"))

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 1
        assert "network unreachable" in capsys.readouterr().err

    def test_returns_2_on_not_implemented_scaffold_stage(self, monkeypatch, capsys):
        _stub_build_pipeline(monkeypatch, NotImplementedError("FfmpegAudioExtractor is a scaffold"))

        exit_code = main_module.main(MIN_ARGS)

        assert exit_code == 2
        assert "scaffold" in capsys.readouterr().err

    def test_missing_required_args_exits_before_reaching_the_pipeline(self, monkeypatch):
        called = False

        def fail_if_called(config):
            nonlocal called
            called = True

        monkeypatch.setattr(main_module, "build_pipeline", fail_if_called)

        with pytest.raises(SystemExit):
            main_module.main(["--url", "https://example.com/v"])  # missing --text

        assert not called
