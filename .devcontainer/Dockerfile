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

# Install Rust toolchain system-wide.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain 1.97.1 \
    --component rustfmt,clippy,llvm-tools-preview

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uvx /usr/local/bin/uvx

# Ensure Rust and uv stay on PATH for login shells (/etc/profile overrides PATH).
RUN echo 'export RUSTUP_HOME=/usr/local/rustup' > /etc/profile.d/rust.sh \
    && echo 'export CARGO_HOME=/usr/local/cargo' >> /etc/profile.d/rust.sh \
    && echo 'export PATH=/usr/local/cargo/bin:/usr/local/bin:$PATH' >> /etc/profile.d/rust.sh

# Create a non-root user for VS Code.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && mkdir -p /workspace \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup /workspace

USER vscode
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/bin:$PATH
WORKDIR /workspace
