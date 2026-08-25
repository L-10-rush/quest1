"""Vosk-backed TranscriptionEngine -- lightweight fallback (see approach.md §4).

Trades some word-timestamp precision for a much smaller model and no GPU
dependency -- use when the dev/eval machine can't comfortably run WhisperX
(`--engine vosk`). Requires a Vosk model directory downloaded separately
(see README.md) -- unlike WhisperX, the `vosk` package doesn't fetch
models on its own.
"""

from __future__ import annotations

import json
import logging
import wave

from src.audio.base import AudioAsset
from src.exceptions import TranscriptionError
from src.transcription.base import DialogueSegment, TranscriptionEngine, TranscriptResult, Word

logger = logging.getLogger(__name__)

# How much audio to feed the recognizer per AcceptWaveform() call. Vosk
# processes streamed chunks internally; this just bounds peak memory for
# a long input rather than reading the whole file at once.
_CHUNK_FRAMES = 4000


class VoskEngine(TranscriptionEngine):
    """Wraps Vosk for a fast, low-footprint alternative to WhisperX."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None  # lazy-loaded: expensive, reused across calls

    def transcribe(self, audio: AudioAsset, language: str) -> TranscriptResult:
        try:
            words, segments = self._transcribe_wav(audio)
        except Exception as exc:
            raise TranscriptionError(f"Vosk transcription failed: {exc}") from exc

        return TranscriptResult(
            words=tuple(words),
            segments=tuple(segments),
            language=language,
            engine_name="vosk",
            model_name=self._model_path,
        )

    def _get_model(self):
        import vosk

        if self._model is None:
            logger.info("loading Vosk model from %s", self._model_path)
            vosk.SetLogLevel(-1)  # suppress Kaldi's very verbose stderr logging
            self._model = vosk.Model(model_path=self._model_path)
        return self._model

    def _transcribe_wav(self, audio: AudioAsset) -> tuple[list[Word], list[DialogueSegment]]:
        import vosk

        with wave.open(str(audio.file_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError(
                    f"Vosk requires mono 16-bit PCM audio, got "
                    f"{wav_file.getnchannels()} channel(s) / "
                    f"{wav_file.getsampwidth() * 8}-bit -- "
                    f"FfmpegAudioExtractor should already guarantee this"
                )

            recognizer = vosk.KaldiRecognizer(self._get_model(), wav_file.getframerate())
            recognizer.SetWords(True)

            words: list[Word] = []
            segments: list[DialogueSegment] = []
            while True:
                chunk = wav_file.readframes(_CHUNK_FRAMES)
                if not chunk:
                    break
                if recognizer.AcceptWaveform(chunk):
                    self._consume_result(recognizer.Result(), words, segments)
            self._consume_result(recognizer.FinalResult(), words, segments)
        return words, segments

    @classmethod
    def _consume_result(
        cls, result_json: str, words: list[Word], segments: list[DialogueSegment]
    ) -> None:
        """Vosk's `Result()`/`FinalResult()` each correspond to one of its
        own endpoint-detected utterances -- exactly one `DialogueSegment`
        per non-empty call, built from the same word entries appended to
        `words`, no separate re-segmentation pass needed."""
        chunk_words = cls._words_from_result(result_json)
        words.extend(chunk_words)
        if chunk_words:
            segments.append(
                DialogueSegment(
                    text=" ".join(w.text for w in chunk_words),
                    start_seconds=chunk_words[0].start_seconds,
                    end_seconds=chunk_words[-1].end_seconds,
                    confidence=sum(w.confidence for w in chunk_words) / len(chunk_words),
                )
            )

    @staticmethod
    def _words_from_result(result_json: str) -> list[Word]:
        payload = json.loads(result_json)
        return [
            Word(
                text=entry["word"],
                start_seconds=entry["start"],
                end_seconds=entry["end"],
                confidence=entry.get("conf", 0.0),
            )
            for entry in payload.get("result", [])
        ]
