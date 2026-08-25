"""WhisperX-backed TranscriptionEngine (default engine, see approach.md §4).

STATUS: scaffold only -- `transcribe()` is intentionally left unimplemented.
Model loading / device selection / alignment wiring needs to be validated
against a real audio file and a real GPU-vs-CPU environment, so it's left
for you. The exact WhisperX call sequence is documented below.
"""

from __future__ import annotations

import logging

from src.audio.base import AudioAsset
from src.exceptions import TranscriptionError
from src.transcription.base import TranscriptionEngine, TranscriptResult, Word

logger = logging.getLogger(__name__)


class WhisperXEngine(TranscriptionEngine):
    """Wraps WhisperX: faster-whisper transcription + wav2vec2 forced alignment.

    Chosen over plain Whisper/faster-whisper because forced alignment gives
    sub-100ms word-level timestamps instead of interpolated ones -- see the
    tradeoff table in approach.md §4 for why that precision matters here.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        self._model_size = model_size
        self._device = device
        self._model = None  # lazy-loaded on first use, see TODO below
        self._align_model = None
        self._align_metadata = None

    def transcribe(self, audio: AudioAsset, language: str) -> TranscriptResult:
        """Transcribe `audio` and word-align the result.

        TODO (not implemented -- fill this in):

        1. Lazily load and cache the WhisperX model on `self._model`
           (loading is expensive; don't reload per call):
               import whisperx
               if self._model is None:
                   compute_type = "int8" if self._device == "cpu" else "float16"
                   self._model = whisperx.load_model(
                       self._model_size, self._device, compute_type=compute_type,
                       language=language,
                   )
        2. Load audio and transcribe:
               audio_array = whisperx.load_audio(str(audio.file_path))
               result = self._model.transcribe(audio_array, language=language)
        3. Load (and cache) the language-specific alignment model, then align:
               if self._align_model is None:
                   self._align_model, self._align_metadata = (
                       whisperx.load_align_model(
                           language_code=result["language"], device=self._device
                       )
                   )
               aligned = whisperx.align(
                   result["segments"], self._align_model, self._align_metadata,
                   audio_array, self._device,
               )
        4. Flatten `aligned["word_segments"]` into a tuple of `Word(...)`,
           using each word's `"start"`, `"end"`, and `"score"` (WhisperX's
           per-word confidence) -- skip/log any word missing a timestamp
           rather than crashing (alignment occasionally drops a word).
        5. Wrap steps 1-4 in try/except and raise
           `TranscriptionError(str(exc)) from exc` on failure.
        6. Return `TranscriptResult(words=tuple(words), language=result["language"],
           engine_name="whisperx", model_name=self._model_size)`.

        Note: if a dedicated wav2vec2 aligner isn't available for
        `language`, `whisperx.load_align_model` raises -- catch that
        specific case and fall back to the *unaligned* segment-level
        timestamps (documented graceful degradation, see approach.md §7)
        rather than failing the whole pipeline.
        """
        raise NotImplementedError(
            "WhisperXEngine.transcribe() is a scaffold -- implement steps 1-6 "
            "in the docstring above and validate against a real audio file."
        )
