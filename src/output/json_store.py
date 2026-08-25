"""Filesystem-backed ResultStore.

Fully implemented: pure file I/O (json + cv2.imwrite), independently
testable with a tmp_path fixture -- no network, no ML model.

Layout per video (one video may be queried with several different target
phrases over time; each run gets its own numbered result so nothing is
silently overwritten):

    output/
      <video_id>/
        <video_id>.meta.json        # video-level info, written once, reused
        frames/
          frame_<frame_number>.png  # one saved image per run
        results/
          result_<frame_number>.json  # one JSON report per run
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.exceptions import ResultPersistenceError
from src.frame_locator.base import FrameResult
from src.ingestion.base import VideoMetadata
from src.matching.base import MatchResult
from src.metrics.transcript_metrics import TranscriptMetrics
from src.output.base import ResultStore
from src.transcription.base import TranscriptResult
from src.utils.timestamp import format_timestamp

logger = logging.getLogger(__name__)


class JsonResultStore(ResultStore):
    """Writes `result.json` + `frames/frame_<n>.png` under `output_dir/<video_id>/`."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def save(
        self,
        video: VideoMetadata,
        target_text: str,
        match: MatchResult,
        frame: FrameResult,
        metrics: TranscriptMetrics,
        transcript: TranscriptResult,
    ) -> Path:
        if match.best is None:
            raise ResultPersistenceError(
                "cannot persist a MatchResult with best=None -- matchers must "
                "always return a best-effort candidate (see matching/base.py)"
            )

        video_dir = self._output_dir / video.video_id
        frames_dir = video_dir / "frames"
        results_dir = video_dir / "results"

        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            self._write_video_meta(video_dir, video)

            image_path = frames_dir / f"frame_{frame.frame_number}.png"
            if not cv2.imwrite(str(image_path), frame.image):
                raise ResultPersistenceError(f"cv2.imwrite failed for {image_path}")

            result_path = results_dir / f"result_{frame.frame_number}.json"
            payload = self._build_payload(
                video, target_text, match, frame, metrics, transcript, image_path
            )
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ResultPersistenceError(str(exc)) from exc

        logger.info("wrote result to %s", result_path)
        return result_path

    def _write_video_meta(self, video_dir: Path, video: VideoMetadata) -> None:
        meta_path = video_dir / f"{video.video_id}.meta.json"
        if meta_path.exists():
            return  # video-level metadata doesn't change between runs
        meta_path.write_text(
            json.dumps(self._video_to_dict(video), indent=2), encoding="utf-8"
        )

    @staticmethod
    def _video_to_dict(video: VideoMetadata) -> dict:
        data = asdict(video)
        data["file_path"] = str(video.file_path)
        return data

    def _build_payload(
        self,
        video: VideoMetadata,
        target_text: str,
        match: MatchResult,
        frame: FrameResult,
        metrics: TranscriptMetrics,
        transcript: TranscriptResult,
        image_path: Path,
    ) -> dict:
        best = match.best
        return {
            "video": self._video_to_dict(video),
            "query": {"target_text": target_text},
            "result": {
                "timestamp": frame.timestamp,
                "frame_number": frame.frame_number,
                "matched_text": best.matched_text,
                "match_score": round(best.score, 2),
                "is_uncertain": match.is_uncertain,
                "uncertainty_reason": match.uncertainty_reason,
                "frame_image_path": str(image_path),
            },
            "candidates": [asdict(c) for c in match.candidates],
            "transcript_metrics": asdict(metrics),
            # Every line of dialogue spoken in the video, independent of
            # `target_text` -- one entry per DialogueSegment (see
            # transcription/base.py), in chronological order.
            "transcript": [self._segment_to_dict(s) for s in transcript.segments],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _segment_to_dict(segment) -> dict:
        return {
            "text": segment.text,
            "start_timestamp": format_timestamp(segment.start_seconds),
            "end_timestamp": format_timestamp(segment.end_seconds),
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "confidence": round(segment.confidence, 3),
        }
