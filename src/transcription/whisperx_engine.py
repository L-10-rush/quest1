"""WhisperX-backed TranscriptionEngine (default engine, see approach.md §4).

Wraps WhisperX: faster-whisper transcription (segment-level) + wav2vec2
forced alignment (word-level). Forced alignment is why this engine exists
instead of plain Whisper/faster-whisper -- see the tradeoff table in
approach.md §4 for why sub-100ms word timestamps matter for "find the
exact frame."
"""

from __future__ import annotations

import logging

import whisperx

from src.audio.base import AudioAsset
from src.exceptions import TranscriptionError
from src.transcription.base import TranscriptionEngine, TranscriptResult, Word

logger = logging.getLogger(__name__)

# WhisperX's default VAD backend ("pyannote") needs a HuggingFace auth
# token for a gated model -- "silero" (Silero VAD, via torch.hub, publicly
# downloadable) needs no account/token, which matters for the "works
# without manual setup" requirement in the problem statement.
_VAD_METHOD = "silero"

# whisperx's own CLI default (whisperx/__main__.py) -- a reasonable
# batch size for both CPU and small-GPU inference.
_DEFAULT_BATCH_SIZE = 8


class WhisperXEngine(TranscriptionEngine):
    """Wraps WhisperX: faster-whisper transcription + wav2vec2 forced alignment."""

    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        self._model_size = model_size
        self._device = device
        self._model = None  # lazy-loaded: expensive, reused across calls
        self._align_models: dict[str, tuple] = {}  # language -> (model, metadata)

    def transcribe(self, audio: AudioAsset, language: str) -> TranscriptResult:
        try:
            model = self._get_model()
            audio_array = whisperx.load_audio(str(audio.file_path))
            result = model.transcribe(
                audio_array, batch_size=_DEFAULT_BATCH_SIZE, language=language
            )
            detected_language = result.get("language", language)

            words = self._align(result["segments"], audio_array, detected_language)
        except Exception as exc:
            # WhisperX/faster-whisper/torch each raise their own exception
            # types depending on the failure -- normalized to
            # TranscriptionError per the interface contract (transcription/base.py).
            raise TranscriptionError(f"WhisperX transcription failed: {exc}") from exc

        return TranscriptResult(
            words=tuple(words),
            language=detected_language,
            engine_name="whisperx",
            model_name=self._model_size,
        )

    def _get_model(self):
        if self._model is None:
            compute_type = "int8" if self._device == "cpu" else "float16"
            logger.info(
                "loading WhisperX model %r on %s (compute_type=%s)",
                self._model_size,
                self._device,
                compute_type,
            )
            self._model = whisperx.load_model(
                self._model_size,
                self._device,
                compute_type=compute_type,
                vad_method=_VAD_METHOD,
            )
        return self._model

    def _align(self, segments: list[dict], audio_array, language: str) -> list[Word]:
        """Word-align `segments` against the audio, or fall back to
        interpolated per-word timestamps if no aligner exists for
        `language` (documented graceful degradation, see approach.md §7)."""
        if not segments:
            return []

        try:
            align_model, align_metadata = self._get_align_model(language)
        except ValueError:
            logger.warning(
                "no wav2vec2 alignment model available for language=%r -- "
                "falling back to interpolated (less precise) word timestamps",
                language,
            )
            return self._interpolate_words(segments)

        aligned = whisperx.align(
            segments, align_model, align_metadata, audio_array, self._device
        )
        return self._words_from_aligned_segments(aligned["word_segments"])

    def _get_align_model(self, language: str):
        if language not in self._align_models:
            logger.info("loading wav2vec2 alignment model for language=%r", language)
            self._align_models[language] = whisperx.load_align_model(
                language_code=language, device=self._device
            )
        return self._align_models[language]

    @staticmethod
    def _words_from_aligned_segments(word_segments: list[dict]) -> list[Word]:
        words: list[Word] = []
        for w in word_segments:
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                # Alignment occasionally can't place a word (e.g. bare
                # punctuation) -- skip it rather than fabricate a timestamp.
                logger.debug("skipping word with no aligned timestamp: %r", w)
                continue
            words.append(
                Word(text=w["word"], start_seconds=start, end_seconds=end, confidence=w.get("score", 0.0))
            )
        return words

    @staticmethod
    def _interpolate_words(segments: list[dict]) -> list[Word]:
        """Split each segment's text evenly across its [start, end] span,
        proportional to word length. Materially less precise than forced
        alignment (~1s drift is possible, see approach.md §4's tradeoff
        table for the Whisper-alone row) -- used only when no aligner
        exists for the detected language."""
        words: list[Word] = []
        for seg in segments:
            tokens = seg["text"].split()
            if not tokens:
                continue
            span = seg["end"] - seg["start"]
            total_chars = sum(len(t) for t in tokens) or 1
            cursor = seg["start"]
            for token in tokens:
                duration = span * (len(token) / total_chars)
                words.append(
                    Word(
                        text=token,
                        start_seconds=cursor,
                        end_seconds=cursor + duration,
                        # Lower than a real alignment score -- signals to
                        # the matcher's low-confidence check (approach.md §6)
                        # that these timestamps are interpolated, not measured.
                        confidence=0.5,
                    )
                )
                cursor += duration
        return words
