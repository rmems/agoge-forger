"""Launch vLLM servers."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys

from ..logging import logger
from .config import ServingConfig
from .diagnostics import get_vllm_version


def build_serve_command(cfg: ServingConfig) -> list[str]:
    """Build the `python -m vllm serve ...` argument list."""
    cmd = [
        sys.executable,
        "-m",
        "vllm",
        "serve",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
    ]
    if cfg.max_model_len is not None:
        cmd.extend(["--max-model-len", str(cfg.max_model_len)])
    if cfg.dtype is not None:
        cmd.extend(["--dtype", cfg.dtype])
    if cfg.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)])
    cmd.extend(cfg.extra_args)
    return cmd


def serve_vllm(cfg: ServingConfig) -> int:
    """Launch a vLLM server.

    Returns an exit code: 0 for success/dry-run, 1 for a hard failure.
    """
    cmd = build_serve_command(cfg)

    if cfg.dry_run:
        logger.info("[dry-run] would execute:")
        logger.info(" ".join(cmd))
        return 0

    version = get_vllm_version()
    if version is None:
        logger.error("vLLM is not installed. Install vLLM to use serve-vllm.")
        return 1

    logger.info("Starting vLLM: %s", " ".join(cmd))
    serve_env = os.environ.copy()
    serve_env["VLLM_USE_RUST_FRONTEND"] = "0"
    try:
        proc = subprocess.run(  # nosec B603  # nosemgrep
            cmd,
            check=False,
            shell=False,
            env=serve_env,
        )
    except FileNotFoundError:
        logger.error("vLLM entry point not found. Is vLLM installed?")
        return 1
    return proc.returncode
