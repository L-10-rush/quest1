"""Unit tests for WhisperXEngine's orchestration.

WhisperX itself (model loading, VAD, forced alignment) is mocked out at
the module boundary -- these tests verify OUR code: lazy model/aligner
caching, mapping WhisperX's word-segment dicts to `Word`, skipping words
with no aligned timestamp, the interpolated-timestamp fallback when no
aligner exists for a language, and error wrapping. No network, no model
download, no audio file needed.
"""

from pathlib import Path

import pytest

import src.transcription.whisperx_engine as whisperx_engine_module
from src.audio.base import AudioAsset
from src.exceptions import TranscriptionError
from src.transcription.whisperx_engine import WhisperXEngine

AUDIO = AudioAsset(file_path=Path("audio.wav"), sample_rate_hz=16_000, channels=1, duration_seconds=2.0)


class FakeModel:
    """Mimics WhisperX's FasterWhisperPipeline.transcribe(): echoes back
    whichever language it was called with, the way the real pipeline's
    result reflects the language actually used for that call."""

    def __init__(self, segments):
        self.calls = []
        self._segments = segments

    def transcribe(self, audio_array, batch_size=None, language=None):
        self.calls.append({"language": language, "batch_size": batch_size})
        return {"segments": self._segments, "language": language}


@pytest.fixture(autouse=True)
def stub_whisperx(monkeypatch):
    """Stub `whisperx.load_audio` (never exercised for real -- these tests
    never touch a real audio file) so every test starts from a clean,
    predictable double without needing to repeat this in each test."""

    def fake_load_audio(path):
        return "fake-audio-array"

    monkeypatch.setattr(whisperx_engine_module.whisperx, "load_audio", fake_load_audio)


class TestTranscribe:
    def test_returns_words_from_aligned_segments(self, monkeypatch, stub_whisperx):
        segments = [{"start": 0.0, "end": 1.5, "text": "My mind rebels at stagnation"}]
        fake_model = FakeModel(segments)
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", lambda *a, **k: fake_model)
        monkeypatch.setattr(
            whisperx_engine_module.whisperx, "load_align_model", lambda language_code, device: ("align_model", "align_meta")
        )
        word_segments = [
            {"word": "My", "start": 0.0, "end": 0.2, "score": 0.9},
            {"word": "mind", "start": 0.2, "end": 0.5, "score": 0.95},
        ]
        monkeypatch.setattr(
            whisperx_engine_module.whisperx, "align", lambda segs, m, meta, audio, device: {"word_segments": word_segments}
        )

        result = WhisperXEngine(model_size="tiny", device="cpu").transcribe(AUDIO, language="en")

        assert result.engine_name == "whisperx"
        assert result.model_name == "tiny"
        assert result.language == "en"
        assert len(result.words) == 2
        assert result.words[0].text == "My"
        assert result.words[0].start_seconds == 0.0
        assert result.words[1].confidence == 0.95

    def test_model_is_loaded_once_and_reused_across_calls(self, monkeypatch, stub_whisperx):
        load_model_calls = []

        def fake_load_model(model_size, device, compute_type, vad_method):
            load_model_calls.append((model_size, device, compute_type, vad_method))
            return FakeModel([])

        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", fake_load_model)
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_align_model", lambda **k: ("m", "meta"))
        monkeypatch.setattr(whisperx_engine_module.whisperx, "align", lambda *a, **k: {"word_segments": []})

        engine = WhisperXEngine(model_size="tiny", device="cpu")
        engine.transcribe(AUDIO, language="en")
        engine.transcribe(AUDIO, language="en")

        assert len(load_model_calls) == 1
        assert load_model_calls[0] == ("tiny", "cpu", "int8", "silero")

    def test_cpu_uses_int8_and_cuda_uses_float16_compute_type(self, monkeypatch, stub_whisperx):
        seen_compute_types = []

        def fake_load_model(model_size, device, compute_type, vad_method):
            seen_compute_types.append(compute_type)
            return FakeModel([])

        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", fake_load_model)
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_align_model", lambda **k: ("m", "meta"))
        monkeypatch.setattr(whisperx_engine_module.whisperx, "align", lambda *a, **k: {"word_segments": []})

        WhisperXEngine(device="cpu").transcribe(AUDIO, language="en")
        WhisperXEngine(device="cuda").transcribe(AUDIO, language="en")

        assert seen_compute_types == ["int8", "float16"]

    def test_align_model_cached_per_language_not_shared(self, monkeypatch, stub_whisperx):
        align_calls = []

        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", lambda *a, **k: FakeModel([{"start": 0, "end": 1, "text": "hi"}]))

        def fake_load_align_model(language_code, device):
            align_calls.append(language_code)
            return (f"model-{language_code}", f"meta-{language_code}")

        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_align_model", fake_load_align_model)
        monkeypatch.setattr(whisperx_engine_module.whisperx, "align", lambda *a, **k: {"word_segments": []})

        engine = WhisperXEngine()
        engine.transcribe(AUDIO, language="en")
        engine.transcribe(AUDIO, language="en")  # same language -- no reload
        engine.transcribe(AUDIO, language="ja")  # different language -- reload

        assert align_calls == ["en", "ja"]

    def test_words_missing_aligned_timestamp_are_skipped(self, monkeypatch, stub_whisperx):
        segments = [{"start": 0.0, "end": 1.0, "text": "um hello"}]
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", lambda *a, **k: FakeModel(segments))
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_align_model", lambda **k: ("m", "meta"))
        word_segments = [
            {"word": "um"},  # alignment couldn't place this one -- no start/end
            {"word": "hello", "start": 0.3, "end": 0.8, "score": 0.9},
        ]
        monkeypatch.setattr(whisperx_engine_module.whisperx, "align", lambda *a, **k: {"word_segments": word_segments})

        result = WhisperXEngine().transcribe(AUDIO, language="en")

        assert len(result.words) == 1
        assert result.words[0].text == "hello"

    def test_falls_back_to_interpolated_words_when_no_aligner_for_language(
        self, monkeypatch, stub_whisperx
    ):
        segments = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", lambda *a, **k: FakeModel(segments))

        def raise_value_error(language_code, device):
            raise ValueError(f"No default align-model for language: {language_code}")

        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_align_model", raise_value_error)

        result = WhisperXEngine().transcribe(AUDIO, language="xx")

        assert len(result.words) == 2
        assert result.words[0].text == "hello"
        assert result.words[0].confidence == 0.5  # marked as lower-confidence, interpolated
        # words should span the segment in order without gaps
        assert result.words[0].start_seconds == 0.0
        assert result.words[1].end_seconds == pytest.approx(2.0)

    def test_empty_segments_produce_empty_transcript_without_calling_align(
        self, monkeypatch, stub_whisperx
    ):
        align_called = []
        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", lambda *a, **k: FakeModel([]))
        monkeypatch.setattr(whisperx_engine_module.whisperx, "align", lambda *a, **k: align_called.append(1))

        result = WhisperXEngine().transcribe(AUDIO, language="en")

        assert result.words == ()
        assert align_called == []

    def test_model_failure_wrapped_in_transcription_error(self, monkeypatch, stub_whisperx):
        class FailingModel:
            def transcribe(self, *a, **k):
                raise RuntimeError("model exploded")

        monkeypatch.setattr(whisperx_engine_module.whisperx, "load_model", lambda *a, **k: FailingModel())

        with pytest.raises(TranscriptionError, match="model exploded"):
            WhisperXEngine().transcribe(AUDIO, language="en")
