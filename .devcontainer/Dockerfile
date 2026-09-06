FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uvx /usr/local/bin/uvx

# Create a non-root user for VS Code.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && mkdir -p /workspace \
    && chown -R vscode:vscode /workspace

USER vscode
WORKDIR /workspace
