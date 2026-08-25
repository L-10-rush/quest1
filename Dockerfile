# ---- builder ---------------------------------------------------------------
# Resolves and installs dependencies with uv into a venv, kept in a separate
# stage so the final image never carries build tooling or the uv binary.
#
# Deliberately uses no BuildKit-only syntax (no `# syntax=`, no
# `RUN --mount=type=cache`) so this builds with the classic Docker builder
# too, not just one with the `buildx` plugin installed -- portability over
# a rebuild-speed optimization that isn't available everywhere.
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first, from the lockfile only -- this layer only
# invalidates when pyproject.toml/uv.lock change, not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the actual source and finish the sync (installs the project
# itself, if it were packaged; here it's a no-op beyond re-validating).
COPY src/ ./src
RUN uv sync --frozen --no-dev

# ctranslate2 (a WhisperX dependency) ships a shared library whose
# GNU_STACK ELF segment is marked executable (RWE) -- almost certainly a
# build-toolchain artifact, not something its code actually needs. On a
# kernel that enforces non-executable stacks (several hardened kernel
# configs, e.g. CachyOS's default) that fails at import time with
# `OSError: ... cannot enable executable stack as shared object requires`.
# The host kernel's policy applies inside the container too (Linux
# containers share the host kernel, they don't virtualize it), so this is
# fixed here, once, at build time rather than left as a per-host surprise.
RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && find /app/.venv -iname 'libctranslate2*.so*' -exec patchelf --clear-execstack {} \;

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# ffmpeg is a hard runtime dependency (audio extraction, stage 2) and is
# not installable via pip -- must come from the system package manager.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (security best practice) with a fixed UID so
# bind-mounted ./work and ./output keep predictable ownership on the host.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/work /app/output && chown -R appuser:appuser /app/work /app/output

USER appuser
VOLUME ["/app/work", "/app/output"]

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
