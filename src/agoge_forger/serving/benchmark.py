from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..logging import logger
from ..path_safety import resolve_output_directory
from ..providers.chat_completions import ChatCompletionsClient, ChatCompletionsConfig
from .config import BenchmarkConfig, Frontend, PromptSet, load_prompt_set, to_serving_config
from .diagnostics import (
    get_cuda_version,
    get_environment,
    get_gpu_name,
    get_vllm_version,
    has_rust_frontend_support,
)
from .serve import _set_frontend_env, build_serve_command


@dataclass
class BenchmarkResult:
    run: str
    frontend: str
    stream: bool
    prompt: str
    latency_ms: float
    time_to_first_token_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_sec: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _wait_for_health(
    host: str, port: int, timeout_s: float = 120.0, interval_s: float = 1.0
) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Health check not ready: %s", exc)
        time.sleep(interval_s)
    return False


def _simulate_results(
    cfg: BenchmarkConfig, prompt_set: PromptSet, frontend: Frontend
) -> list[BenchmarkResult]:
    """Deterministic fake results used for dry-run validation without a GPU."""
    results: list[BenchmarkResult] = []
    base_latency = 120.0
    for prompt in prompt_set.prompts:
        n = len(prompt)
        latency = base_latency + n * 0.5
        if frontend == Frontend.rust:
            latency *= 0.85
        if cfg.stream:
            latency *= 0.9
        time_to_first_token_ms = latency * 0.25
        input_tokens = max(1, n // 4)
        output_tokens = 50
        total_tokens = input_tokens + output_tokens
        tokens_per_sec = output_tokens / (latency / 1000.0) if latency > 0 else 0.0
        results.append(
            BenchmarkResult(
                run=cfg.run_name,
                frontend=frontend.value,
                stream=cfg.stream,
                prompt=prompt,
                latency_ms=latency,
                time_to_first_token_ms=time_to_first_token_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                tokens_per_sec=tokens_per_sec,
            )
        )
    return results


@contextmanager
def _managed_server(cfg: BenchmarkConfig, frontend: Frontend):
    """Start a vLLM server for the duration of the context manager."""
    serving_cfg = to_serving_config(cfg, frontend)
    env = _set_frontend_env(os.environ, frontend)
    cmd = build_serve_command(serving_cfg)
    logger.info(f"Starting server for {frontend.value} benchmark: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not _wait_for_health(cfg.host, cfg.port):
            raise RuntimeError(f"vLLM server ({frontend.value} frontend) failed to become healthy")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _run_client_requests(
    cfg: BenchmarkConfig, prompt_set: PromptSet, frontend: Frontend, out_dir: Path
) -> list[BenchmarkResult]:
    base_url = f"http://{cfg.host}:{cfg.port}/v1"
    client = ChatCompletionsClient(
        ChatCompletionsConfig(
            base_url=base_url,
            model=cfg.model,
            stream=cfg.stream,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout_s=120.0,
        ),
        run_name=str(out_dir / f"client_{frontend.value}"),
    )

    results: list[BenchmarkResult] = []
    for prompt in prompt_set.prompts:
        result = client.chat_simple(prompt, system=prompt_set.system, stream=cfg.stream)
        latency = result.latency_ms
        tokens_per_sec = result.output_tokens / (latency / 1000.0) if latency > 0 else 0.0
        results.append(
            BenchmarkResult(
                run=cfg.run_name,
                frontend=frontend.value,
                stream=cfg.stream,
                prompt=prompt,
                latency_ms=latency,
                time_to_first_token_ms=result.time_to_first_token_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                tokens_per_sec=tokens_per_sec,
                error=result.error,
            )
        )
    return results


def _measure_frontend_real(
    cfg: BenchmarkConfig, prompt_set: PromptSet, frontend: Frontend, out_dir: Path
) -> list[BenchmarkResult]:
    try:
        with _managed_server(cfg, frontend):
            return _run_client_requests(cfg, prompt_set, frontend, out_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Benchmark failed for {frontend.value} frontend: {exc}")
        return [
            BenchmarkResult(
                run=cfg.run_name,
                frontend=frontend.value,
                stream=cfg.stream,
                prompt="",
                latency_ms=0.0,
                time_to_first_token_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                tokens_per_sec=0.0,
                error=str(exc),
            )
        ]


def _aggregate(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, bool], list[BenchmarkResult]] = {}
    for r in results:
        groups.setdefault((r.frontend, r.stream), []).append(r)

    rows: list[dict[str, Any]] = []
    for (frontend, stream), group in sorted(groups.items()):
        valid = [r for r in group if not r.error]
        rows.append(
            {
                "frontend": frontend,
                "stream": stream,
                "mean_latency_ms": _mean([r.latency_ms for r in valid]),
                "mean_ttft_ms": _mean([r.time_to_first_token_ms for r in valid]),
                "mean_input_tokens": _mean([float(r.input_tokens) for r in valid]),
                "mean_output_tokens": _mean([float(r.output_tokens) for r in valid]),
                "mean_total_tokens": _mean([float(r.total_tokens) for r in valid]),
                "mean_tokens_per_sec": _mean([r.tokens_per_sec for r in valid]),
            }
        )
    return rows


def _build_summary(out_dir: Path, results: list[BenchmarkResult], cfg: BenchmarkConfig) -> str:
    env = get_environment()
    vllm_version = get_vllm_version() or "not installed"
    cuda_version = get_cuda_version()
    gpu_name = get_gpu_name()
    timestamp = datetime.now(timezone.utc).isoformat()
    aggregate = _aggregate(results)

    lines = [
        "# vLLM Frontend Benchmark Summary",
        "",
        f"**Run:** `{cfg.run_name}`",
        f"**Output:** `{out_dir}`",
        f"**Timestamp:** {timestamp}",
        "",
        "## Environment",
        "",
        f"- **vLLM version:** {vllm_version}",
        f"- **CUDA version:** {cuda_version}",
        f"- **GPU name:** {gpu_name}",
        "- **Environment variables:**",
        "",
        "```json",
        json.dumps(env, indent=2, default=str),
        "```",
        "",
        "## Per-result details",
        "",
        "| Run | Frontend | Stream | Prompt | Latency (ms) | TTFT (ms) | Input | Output | Total | tokens/sec |",
        "|-----|----------|--------|--------|-------------:|----------:|------:|-------:|------:|-----------:|",
    ]
    for r in results:
        prompt = r.prompt.replace("|", "\\|")
        lines.append(
            f"| {r.run} | {r.frontend} | {r.stream} | {prompt} | {r.latency_ms:.2f} | "
            f"{r.time_to_first_token_ms:.2f} | {r.input_tokens} | {r.output_tokens} | "
            f"{r.total_tokens} | {r.tokens_per_sec:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Python vs Rust comparison",
            "",
            "| Frontend | Mean latency (ms) | Mean TTFT (ms) | Mean input tokens | Mean output tokens | Mean total tokens | Mean tokens/sec |",
            "|----------|------------------:|---------------:|------------------:|-------------------:|------------------:|-----------------:|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['frontend']} | {row['mean_latency_ms']:.2f} | {row['mean_ttft_ms']:.2f} | "
            f"{row['mean_input_tokens']:.1f} | {row['mean_output_tokens']:.1f} | "
            f"{row['mean_total_tokens']:.1f} | {row['mean_tokens_per_sec']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def _write_artifacts(out_dir: Path, results: list[BenchmarkResult], cfg: BenchmarkConfig) -> None:
    with open(out_dir / "results.jsonl", "w") as f:
        f.writelines(json.dumps(r.to_dict(), default=str) + "\n" for r in results)

    aggregate = _aggregate(results)
    fieldnames = [
        "frontend",
        "stream",
        "mean_latency_ms",
        "mean_ttft_ms",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "mean_tokens_per_sec",
    ]
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate:
            writer.writerow(row)

    with open(out_dir / "summary.md", "w") as f:
        f.write(_build_summary(out_dir, results, cfg))


def benchmark_vllm_frontends(cfg: BenchmarkConfig) -> int:
    """Run a Python-vs-Rust vLLM frontend benchmark and write artifacts.

    Returns an exit code: 0 for success, 1 for a hard failure.
    """
    if not cfg.prompt_set:
        raise ValueError("Prompt set path is required")
    prompt_set = load_prompt_set(cfg.prompt_set)

    if not cfg.out_dir:
        cfg.out_dir = f"runs/vllm_bench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = resolve_output_directory(cfg.out_dir)

    frontends = [cfg.frontend] if cfg.frontend else [Frontend.python, Frontend.rust]
    all_results: list[BenchmarkResult] = []

    for frontend in frontends:
        if not cfg.dry_run:
            if get_vllm_version() is None:
                logger.error("vLLM is not installed. Install vLLM or use --dry-run.")
                return 1
            if frontend == Frontend.rust:
                supported, msg = has_rust_frontend_support()
                if not supported:
                    logger.error(f"Rust frontend not available: {msg}")
                    return 1
            results = _measure_frontend_real(cfg, prompt_set, frontend, out_dir)
        else:
            results = _simulate_results(cfg, prompt_set, frontend)
        all_results.extend(results)

    _write_artifacts(out_dir, all_results, cfg)
    logger.info(f"Benchmark artifacts written to {out_dir}")
    return 0
