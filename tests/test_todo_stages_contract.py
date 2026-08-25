"""Checklist tests for the scaffolded (not-yet-implemented) stages.

These exist so `pytest` has a visible, honest signal of what's left to
build: each currently asserts the stub's documented `NotImplementedError`
contract. As you implement a stage, replace its test here with real
assertions (see the other test_*.py files for the pattern) -- a stage
"graduates" out of this file.
"""

import pytest

from src.audio.ffmpeg_extractor import FfmpegAudioExtractor
from src.ingestion.ytdlp_downloader import YtDlpDownloader
from src.matching.fuzzy_matcher import FuzzyMatcher
from src.transcription.vosk_engine import VoskEngine
from src.transcription.whisperx_engine import WhisperXEngine


def test_ytdlp_downloader_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        YtDlpDownloader().download("https://ok.ru/video/248244667877", dest_dir=None)


def test_ffmpeg_extractor_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        FfmpegAudioExtractor().extract(video=None, dest_dir=None)


def test_whisperx_engine_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        WhisperXEngine().transcribe(audio=None, language="en")


def test_vosk_engine_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        VoskEngine(model_path="models/vosk").transcribe(audio=None, language="en")


def test_fuzzy_matcher_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        FuzzyMatcher().match(transcript=None, target_text="x", threshold=80.0)
