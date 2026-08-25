"""Unit tests for VoskEngine's orchestration.

The `vosk` library itself (model loading, the Kaldi recognizer) is mocked
out -- these tests verify OUR code: chunked WAV reading, JSON-result to
`Word` mapping, model caching, mono/16-bit format validation, and error
wrapping. No real Vosk model download needed.
"""

import json
import sys
import types
import wave
from pathlib import Path

import pytest

from src.audio.base import AudioAsset
from src.exceptions import TranscriptionError
from src.transcription.vosk_engine import VoskEngine


def _write_wav(path: Path, seconds: float = 1.0, channels: int = 1, sample_width: int = 2, rate: int = 16_000):
    n_frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00" * n_frames * sample_width * channels)


class FakeRecognizer:
    """Mimics vosk.KaldiRecognizer: emits one word per AcceptWaveform call
    that returns True, then flushes a final word on FinalResult()."""

    def __init__(self, model, sample_rate, results):
        self.model = model
        self.sample_rate = sample_rate
        self._results = list(results)
        self._words_set = False

    def SetWords(self, enabled):
        self._words_set = enabled

    def AcceptWaveform(self, chunk):
        return bool(self._results)

    def Result(self):
        if self._results:
            return json.dumps({"result": [self._results.pop(0)]})
        return json.dumps({"result": []})

    def FinalResult(self):
        remaining = self._results
        self._results = []
        return json.dumps({"result": remaining})


@pytest.fixture()
def fake_vosk_module(monkeypatch):
    """Install a fake `vosk` module in sys.modules so `import vosk` inside
    VoskEngine's methods resolves to our double instead of the real thing."""
    module = types.ModuleType("vosk")
    module.model_calls = []
    module.recognizer_results = []

    class FakeModel:
        def __init__(self, model_path=None):
            module.model_calls.append(model_path)

    def make_recognizer(model, sample_rate):
        return FakeRecognizer(model, sample_rate, module.recognizer_results)

    module.Model = FakeModel
    module.KaldiRecognizer = make_recognizer
    module.SetLogLevel = lambda level: None

    monkeypatch.setitem(sys.modules, "vosk", module)
    return module


AUDIO_WORDS = [
    {"word": "my", "start": 0.0, "end": 0.2, "conf": 0.9},
    {"word": "mind", "start": 0.2, "end": 0.5, "conf": 0.95},
]


class TestTranscribe:
    def test_returns_words_from_recognizer_results(self, tmp_path, fake_vosk_module):
        wav_path = tmp_path / "audio.wav"
        _write_wav(wav_path)
        fake_vosk_module.recognizer_results = list(AUDIO_WORDS)
        audio = AudioAsset(file_path=wav_path, sample_rate_hz=16_000, channels=1, duration_seconds=1.0)

        result = VoskEngine(model_path="models/vosk").transcribe(audio, language="en")

        assert result.engine_name == "vosk"
        assert result.language == "en"
        assert len(result.words) == 2
        assert result.words[0].text == "my"
        assert result.words[1].confidence == 0.95

    def test_model_loaded_once_and_reused_across_calls(self, tmp_path, fake_vosk_module):
        wav_path = tmp_path / "audio.wav"
        _write_wav(wav_path)
        audio = AudioAsset(file_path=wav_path, sample_rate_hz=16_000, channels=1, duration_seconds=1.0)

        engine = VoskEngine(model_path="models/vosk")
        engine.transcribe(audio, language="en")
        engine.transcribe(audio, language="en")

        assert fake_vosk_module.model_calls == ["models/vosk"]

    def test_stereo_audio_raises_transcription_error(self, tmp_path, fake_vosk_module):
        wav_path = tmp_path / "audio.wav"
        _write_wav(wav_path, channels=2)
        audio = AudioAsset(file_path=wav_path, sample_rate_hz=16_000, channels=2, duration_seconds=1.0)

        with pytest.raises(TranscriptionError, match="mono"):
            VoskEngine(model_path="models/vosk").transcribe(audio, language="en")

    def test_missing_file_raises_transcription_error(self, tmp_path, fake_vosk_module):
        audio = AudioAsset(
            file_path=tmp_path / "does_not_exist.wav", sample_rate_hz=16_000, channels=1, duration_seconds=1.0
        )

        with pytest.raises(TranscriptionError):
            VoskEngine(model_path="models/vosk").transcribe(audio, language="en")

    def test_empty_final_result_yields_no_words(self, tmp_path, fake_vosk_module):
        wav_path = tmp_path / "audio.wav"
        _write_wav(wav_path)
        fake_vosk_module.recognizer_results = []
        audio = AudioAsset(file_path=wav_path, sample_rate_hz=16_000, channels=1, duration_seconds=1.0)

        result = VoskEngine(model_path="models/vosk").transcribe(audio, language="en")

        assert result.words == ()
