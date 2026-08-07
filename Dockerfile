# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.7.9@sha256:563b73ab264117698521303e361fb781a0b421058661b4055750b6c822262d1e /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.7.9@sha256:563b73ab264117698521303e361fb781a0b421058661b4055750b6c822262d1e /uvx /usr/local/bin/uvx

# Create a non-root runtime user and an app directory it can write to.
RUN useradd -m -u 1000 -s /bin/bash app \
    && mkdir -p /app \
    && chown app:app /app

WORKDIR /app

# Copy project metadata and lockfile first so dependency installation can cache.
COPY --chown=app:app pyproject.toml uv.lock README.md LICENSE ./

# Copy the package source and install locked runtime dependencies.
COPY --chown=app:app src ./src

# Copy built-in configs and the sample dataset they reference.
COPY --chown=app:app configs ./configs
COPY --chown=app:app datasets ./datasets

USER app

# Avoid persisting uv's package cache inside the image; rely on the lockfile
# and fail the build if uv.lock is not consistent with pyproject.toml.
ENV UV_NO_CACHE=1
RUN uv venv /app/.venv \
    && uv sync --no-dev --no-editable --locked

ENV UV_NO_CACHE="" \
    PATH="/app/.venv/bin:$PATH"

# HF_TOKEN must be supplied at runtime; it is never baked into the image.
# trust_remote_code defaults to false in code; opt in explicitly via flags/config.

CMD ["agoge", "--help"]
