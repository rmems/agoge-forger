from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping

from ..logging import logger
from .config import Frontend, ServingConfig
from .diagnostics import get_vllm_version, has_rust_frontend_support


def _set_frontend_env(env: Mapping[str, str], frontend: Frontend) -> dict[str, str]:
    """Return a copy of *env* with VLLM_USE_RUST_FRONTEND set for the chosen frontend."""
    env = dict(env)
    env["VLLM_USE_RUST_FRONTEND"] = "1" if frontend == Frontend.rust else "0"
    return env


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
    """Launch a vLLM server with the requested frontend.

    Returns an exit code: 0 for success/dry-run, 1 for a hard failure.
    """
    version = get_vllm_version()
    if version is None:
        logger.error("vLLM is not installed. Install vLLM to use serve-vllm.")
        return 1

    if cfg.frontend == Frontend.rust:
        supported, msg = has_rust_frontend_support()
        if not supported:
            logger.error(f"Rust frontend requested but not available: {msg}")
            return 1

    env = _set_frontend_env(os.environ, cfg.frontend)
    cmd = build_serve_command(cfg)

    if cfg.dry_run:
        logger.info("[dry-run] would execute:")
        logger.info(" ".join(cmd))
        logger.info(f"[dry-run] VLLM_USE_RUST_FRONTEND={env['VLLM_USE_RUST_FRONTEND']}")
        return 0

    logger.info(f"Starting vLLM ({cfg.frontend.value} frontend): {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError:
        logger.error("vLLM entry point not found. Is vLLM installed?")
        return 1
    return proc.returncode
