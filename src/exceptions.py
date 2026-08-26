"""Pipeline-specific exceptions.

Every stage of the pipeline raises one of these (never a bare Exception) so
that `pipeline.py` can catch failures at a known granularity and `main.py`
can map them to useful CLI error messages / exit codes.
"""


class PipelineError(Exception):
    """Base class for all errors raised by the dialogue-finder pipeline."""


class DownloadError(PipelineError):
    """Raised when the source video could not be fetched (stage 1)."""


class AudioExtractionError(PipelineError):
    """Raised when the audio track could not be extracted from the video (stage 2)."""


class TranscriptionError(PipelineError):
    """Raised when speech-to-text / word alignment fails (stage 3)."""


class MatchingError(PipelineError):
    """Raised when the phrase matcher cannot run at all (not the same as "no
    confident match" -- that case is a normal, non-exceptional result with
    ``MatchResult.is_uncertain = True``, see matching/base.py)."""


class FrameExtractionError(PipelineError):
    """Raised when the target frame could not be read from the video (stage 5)."""


class ScreenPresenceError(PipelineError):
    """Raised when on-screen speaker verification cannot run at all (stage 6)
    -- e.g. the video file can't be opened. NOT the same as an inconclusive
    verdict, which is a normal ScreenPresenceResult with status="uncertain"
    (see screen_presence/base.py); that case is never an exception."""


class ResultPersistenceError(PipelineError):
    """Raised when the final result could not be written to disk (stage 7)."""
