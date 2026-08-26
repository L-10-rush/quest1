"""OpenCV-backed ScreenPresenceDetector -- a self-contained heuristic for
whether a speaking face was visible on camera during a matched dialogue
window (see approach.md and README's Known Limitations).

Deliberately NOT a trained active-speaker-detection model. A comparable
project (see approach.md's comparison notes) uses a hosted active-speaker-
detection service plus speaker diarization -- more accurate, but it needs
an API key, a gated model token, and live network access to a third party
at evaluation time. This heuristic trades some of that accuracy for being
fully self-contained: face detection via OpenCV's own bundled Haar
cascade (no extra model download, no network call), and a frame-to-frame
mouth-region motion signal as a proxy for "is this visible face actually
talking" rather than a static shot or a silent listener.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.exceptions import ScreenPresenceError
from src.ingestion.base import VideoMetadata
from src.screen_presence.base import (
    ScreenPresenceDetector,
    ScreenPresenceResult,
    ScreenPresenceStatus,
)

logger = logging.getLogger(__name__)


class OpenCvScreenPresenceDetector(ScreenPresenceDetector):
    """Samples frames across a matched window, detects the largest face
    with OpenCV's bundled Haar cascade, and scores mouth-region motion
    across consecutive detections as a proxy for active speech."""

    #: How many frames to sample across the (padded) window. More samples
    #: make the motion signal steadier but cost more seeks.
    _SAMPLE_COUNT = 8

    #: Extra seconds padded onto both ends of a short match window so a
    #: one-word match still gets enough samples to say anything.
    _PAD_SECONDS = 0.35

    #: Below this face-detection ratio, call it off_screen outright.
    _MIN_FACE_RATIO_UNCERTAIN = 0.15
    #: At or above this ratio (and with enough mouth motion), call it on_screen.
    _MIN_FACE_RATIO_ON_SCREEN = 0.5
    #: Below this normalized mouth-motion score, a consistently-visible face
    #: still isn't confidently "speaking" -- could be a static shot, or a
    #: silent listener rather than the one talking.
    _MIN_MOUTH_MOTION = 0.02
    #: Mouth-motion score at/above this is treated as "clearly moving" for
    #: confidence purposes -- a normalization cap, not a hard cutoff.
    _MOUTH_MOTION_SATURATION = 0.08

    #: Normalizes away face-bounding-box size jitter between frames so the
    #: motion diff compares like-sized crops.
    _MOUTH_CROP_SIZE = (32, 32)

    #: Reasonable defaults, not empirically calibrated against a labeled
    #: dataset (unlike, say, the ASR/matcher thresholds elsewhere in this
    #: project, which were tuned against real transcription runs) -- see
    #: approach.md for the honest scope of what this heuristic is and isn't.

    def __init__(self) -> None:
        self._cascade: cv2.CascadeClassifier | None = None  # lazy: cheap, but no need before first use

    def verify(
        self, video: VideoMetadata, start_seconds: float, end_seconds: float
    ) -> ScreenPresenceResult:
        window_start = max(0.0, start_seconds - self._PAD_SECONDS)
        window_end = end_seconds + self._PAD_SECONDS
        if video.duration_seconds:
            window_end = min(video.duration_seconds, window_end)
        if window_end <= window_start:
            window_end = window_start + self._PAD_SECONDS

        cap = cv2.VideoCapture(str(video.file_path))
        if not cap.isOpened():
            raise ScreenPresenceError(f"could not open video file: {video.file_path}")

        try:
            frames_sampled, mouth_crops = self._sample_mouth_crops(
                cap, video.fps, window_start, window_end
            )
        finally:
            cap.release()

        if frames_sampled == 0:
            return ScreenPresenceResult(
                status="uncertain",
                confidence=0.0,
                reason="could not read any frames in the matched window",
                face_ratio=0.0,
                mouth_motion_score=0.0,
                frames_sampled=0,
            )

        face_ratio = len(mouth_crops) / frames_sampled
        mouth_motion_score = self._motion_score(mouth_crops)
        status, confidence, reason = self._decide(face_ratio, mouth_motion_score)

        return ScreenPresenceResult(
            status=status,
            confidence=confidence,
            reason=reason,
            face_ratio=round(face_ratio, 3),
            mouth_motion_score=round(mouth_motion_score, 4),
            frames_sampled=frames_sampled,
        )

    def _sample_mouth_crops(
        self, cap: cv2.VideoCapture, fps: float, window_start: float, window_end: float
    ) -> tuple[int, list[np.ndarray]]:
        timestamps = np.linspace(window_start, window_end, num=self._SAMPLE_COUNT)
        frames_sampled = 0
        crops: list[np.ndarray] = []
        for t in timestamps:
            frame = self._read_frame_at(cap, float(t), fps)
            if frame is None:
                continue  # a missed seek/read isn't evidence of "no face" -- just skip it
            frames_sampled += 1
            crop = self._detect_mouth_crop(frame)
            if crop is not None:
                crops.append(crop)
        return frames_sampled, crops

    @staticmethod
    def _read_frame_at(cap: cv2.VideoCapture, timestamp_seconds: float, fps: float) -> np.ndarray | None:
        # A coarse presence sample doesn't need OpenCvFrameLocator's
        # drift-tolerant exact-frame guarantee -- missing a sample by a
        # frame or two doesn't change the aggregate face/motion signal, so
        # a plain direct seek is enough here.
        frame_number = round(timestamp_seconds * fps) if fps else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, frame_number)))
        ok, frame = cap.read()
        return frame if ok else None

    def _detect_mouth_crop(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None

        # Largest detected face -- the one most likely to be the framed subject.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mouth_top = y + int(h * 0.55)  # lower ~45% of the face box: mouth/chin region
        mouth = gray[mouth_top : y + h, x : x + w]
        if mouth.size == 0:
            return None
        return cv2.resize(mouth, self._MOUTH_CROP_SIZE)

    def _get_cascade(self) -> cv2.CascadeClassifier:
        if self._cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
        return self._cascade

    @staticmethod
    def _motion_score(crops: list[np.ndarray]) -> float:
        if len(crops) < 2:
            return 0.0
        diffs = [
            float(np.mean(cv2.absdiff(crops[i], crops[i + 1]))) / 255.0
            for i in range(len(crops) - 1)
        ]
        return sum(diffs) / len(diffs)

    @classmethod
    def _decide(
        cls, face_ratio: float, mouth_motion_score: float
    ) -> tuple[ScreenPresenceStatus, float, str]:
        if face_ratio < cls._MIN_FACE_RATIO_UNCERTAIN:
            return (
                "off_screen",
                round(1.0 - face_ratio, 3),
                f"no face detected in {face_ratio:.0%} of sampled frames",
            )
        if face_ratio < cls._MIN_FACE_RATIO_ON_SCREEN:
            return (
                "uncertain",
                round(face_ratio, 3),
                f"a face was visible in only {face_ratio:.0%} of sampled frames",
            )
        if mouth_motion_score < cls._MIN_MOUTH_MOTION:
            return (
                "uncertain",
                round(face_ratio * 0.5, 3),
                (
                    "a face was consistently visible but showed little mouth "
                    "movement -- may be a static shot or a silent listener, "
                    "not necessarily the speaker"
                ),
            )
        confidence = min(
            1.0, 0.5 * face_ratio + 0.5 * min(1.0, mouth_motion_score / cls._MOUTH_MOTION_SATURATION)
        )
        return (
            "on_screen",
            round(confidence, 3),
            (
                f"a face was visible in {face_ratio:.0%} of sampled frames with "
                f"mouth movement consistent with speech"
            ),
        )
