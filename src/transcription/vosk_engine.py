"""Vosk-backed TranscriptionEngine -- lightweight fallback (see approach.md §4).

STATUS: scaffold only, same reasoning as whisperx_engine.py: needs a real
audio file + downloaded Vosk model to validate against.
"""

from __future__ import annotations

import logging

from src.audio.base import AudioAsset
from src.exceptions import TranscriptionError
from src.transcription.base import TranscriptionEngine, TranscriptResult, Word

logger = logging.getLogger(__name__)


class VoskEngine(TranscriptionEngine):
    """Wraps Vosk for a fast, low-footprint alternative to WhisperX.

    Trades some word-timestamp precision for a much smaller model and no
    GPU dependency -- use when the dev/eval machine can't comfortably run
    WhisperX (`--engine vosk`).
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None  # lazy-loaded, see TODO below

    def transcribe(self, audio: AudioAsset, language: str) -> TranscriptResult:
        """Transcribe `audio` using a local Vosk model.

        TODO (not implemented -- fill this in):

        1. Lazily load and cache the model:
               import vosk
               if self._model is None:
                   self._model = vosk.Model(self._model_path)
        2. Open `audio.file_path` with the stdlib `wave` module (Vosk
           expects raw PCM16 mono -- `FfmpegAudioExtractor` already
           produces exactly that format) and feed it through
           `vosk.KaldiRecognizer(self._model, audio.sample_rate_hz)` in
           chunks, calling `.AcceptWaveform(chunk)` per chunk and
           `.FinalResult()` at the end.
        3. Vosk returns JSON with a `"result"` list of
           `{"word", "start", "end", "conf"}` per word when
           `SetWords(True)` is enabled on the recognizer -- map each entry
           to a `Word(text=..., start_seconds=..., end_seconds=...,
           confidence=...)`.
        4. Wrap steps 1-3 in try/except and raise
           `TranscriptionError(str(exc)) from exc` on failure.
        5. Return `TranscriptResult(words=tuple(words), language=language,
           engine_name="vosk", model_name=self._model_path)`.
        """
        raise NotImplementedError(
            "VoskEngine.transcribe() is a scaffold -- implement steps 1-5 in "
            "the docstring above and validate against a real audio file."
        )
