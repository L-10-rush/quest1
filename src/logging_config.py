"""Central logging setup.

Kept as one function so every entrypoint (CLI, tests, a future API wrapper)
configures logging identically instead of each module calling
``logging.basicConfig`` with slightly different formats.
"""

import logging
import sys


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger once, for the whole process.

    Args:
        verbose: when True, sets DEBUG level and includes module names;
            otherwise INFO level with a terse message-only format.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if verbose
        else "%(asctime)s | %(levelname)-8s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)

    # Third-party libraries (whisperx/torch/urllib3) are extremely chatty at
    # INFO/DEBUG; keep them at WARNING unless we're explicitly debugging.
    if not verbose:
        for noisy in ("urllib3", "torch", "speechbrain", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
