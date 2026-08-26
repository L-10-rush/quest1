"""Unit tests for PipelineConfig validation and CLI argument parsing.

Sprint 3 (robustness): proves the config knobs called out in approach.md
(match threshold, window size, --language) are real, validated, and wired
correctly -- and that bad input fails loudly (ValueError / SystemExit) at
construction time rather than surfacing as a confusing failure three
stages deep into a run.
"""

from pathlib import Path

import pytest

from src.config import (
    DEFAULT_ENGINE,
    DEFAULT_LANGUAGE,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
    PipelineConfig,
    config_from_args,
)

MIN_ARGS = ["--url", "https://ok.ru/video/248244667877", "--text", "My mind rebels at stagnation"]


class TestPipelineConfigValidation:
    def test_valid_config_constructs(self):
        config = PipelineConfig(video_url="https://example.com/v", target_text="hello world")
        assert config.language == DEFAULT_LANGUAGE

    def test_empty_video_url_raises(self):
        with pytest.raises(ValueError, match="video_url"):
            PipelineConfig(video_url="", target_text="hello")

    def test_whitespace_only_video_url_raises(self):
        with pytest.raises(ValueError, match="video_url"):
            PipelineConfig(video_url="   ", target_text="hello")

    def test_empty_target_text_raises(self):
        with pytest.raises(ValueError, match="target_text"):
            PipelineConfig(video_url="https://example.com/v", target_text="")

    @pytest.mark.parametrize("threshold", [-0.1, 100.1, -50, 1000])
    def test_match_threshold_out_of_range_raises(self, threshold):
        with pytest.raises(ValueError, match="match_threshold"):
            PipelineConfig(
                video_url="https://example.com/v", target_text="hi", match_threshold=threshold
            )

    @pytest.mark.parametrize("threshold", [0, 100, 0.0, 100.0, 50.5])
    def test_match_threshold_boundary_values_are_valid(self, threshold):
        config = PipelineConfig(
            video_url="https://example.com/v", target_text="hi", match_threshold=threshold
        )
        assert config.match_threshold == threshold

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="engine"):
            PipelineConfig(video_url="https://example.com/v", target_text="hi", engine="bogus")

    @pytest.mark.parametrize("engine", ["whisperx", "vosk"])
    def test_known_engines_are_valid(self, engine):
        config = PipelineConfig(video_url="https://example.com/v", target_text="hi", engine=engine)
        assert config.engine == engine

    def test_config_is_frozen(self):
        config = PipelineConfig(video_url="https://example.com/v", target_text="hi")
        with pytest.raises(AttributeError):
            config.target_text = "changed"


class TestConfigFromArgs:
    def test_requires_url(self):
        with pytest.raises(SystemExit):
            config_from_args(["--text", "hello"])

    def test_text_defaults_to_none_when_omitted(self):
        """`--text` is optional -- omitting it signals main.py to start an
        interactive session instead of failing argument parsing."""
        config = config_from_args(["--url", "https://example.com/v"])
        assert config.target_text is None

    def test_defaults_applied(self):
        config = config_from_args(MIN_ARGS)
        assert config.language == DEFAULT_LANGUAGE
        assert config.engine == DEFAULT_ENGINE
        assert config.whisper_model == DEFAULT_WHISPER_MODEL
        assert config.match_threshold == DEFAULT_MATCH_THRESHOLD
        assert config.window_size is None
        assert config.work_dir == Path("work")
        assert config.output_dir == Path("output")
        assert config.keep_work_files is False
        assert config.verbose is False

    def test_language_flag_is_demonstrably_swappable(self):
        """The exit criterion from approach.md Sprint 3: changing
        `--language` requires no pipeline code change, just a flag."""
        english = config_from_args([*MIN_ARGS, "--language", "en"])
        spanish = config_from_args([*MIN_ARGS, "--language", "es"])
        japanese = config_from_args([*MIN_ARGS, "--language", "ja"])

        assert english.language == "en"
        assert spanish.language == "es"
        assert japanese.language == "ja"

    def test_match_threshold_flag_parsed_as_float(self):
        config = config_from_args([*MIN_ARGS, "--match-threshold", "65.5"])
        assert config.match_threshold == 65.5

    def test_invalid_match_threshold_flag_raises_value_error(self):
        # Parses fine as a float (argparse), but fails PipelineConfig's
        # own range validation -- proves the two layers compose correctly.
        with pytest.raises(ValueError, match="match_threshold"):
            config_from_args([*MIN_ARGS, "--match-threshold", "150"])

    def test_window_size_flag_parsed_as_int(self):
        config = config_from_args([*MIN_ARGS, "--window-size", "7"])
        assert config.window_size == 7

    def test_engine_flag_rejects_unknown_choice(self):
        with pytest.raises(SystemExit):
            config_from_args([*MIN_ARGS, "--engine", "bogus"])

    def test_keep_work_files_flag(self):
        assert config_from_args(MIN_ARGS).keep_work_files is False
        assert config_from_args([*MIN_ARGS, "--keep-work-files"]).keep_work_files is True

    def test_verbose_flag(self):
        assert config_from_args(MIN_ARGS).verbose is False
        assert config_from_args([*MIN_ARGS, "--verbose"]).verbose is True

    def test_work_dir_and_output_dir_flags(self):
        config = config_from_args(
            [*MIN_ARGS, "--work-dir", "/tmp/scratch", "--output-dir", "/tmp/results"]
        )
        assert config.work_dir == Path("/tmp/scratch")
        assert config.output_dir == Path("/tmp/results")

    def test_env_var_used_as_default_when_no_flag_given(self, monkeypatch):
        monkeypatch.setenv("LANGUAGE", "fr")
        monkeypatch.setenv("MATCH_THRESHOLD", "90")
        config = config_from_args(MIN_ARGS)
        assert config.language == "fr"
        assert config.match_threshold == 90.0

    def test_cli_flag_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("LANGUAGE", "fr")
        config = config_from_args([*MIN_ARGS, "--language", "de"])
        assert config.language == "de"
