"""vLLM compatibility smoke test using the OpenAI chat-completions client.

This module provides a lightweight entrypoint for verifying that a model
exposed by a vLLM/OpenAI-compatible endpoint responds to a minimal chat
completion. It reuses `ChatCompletionsClient` and writes structured results
under `runs/<run_name>/`.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging import logger
from ..path_safety import resolve_output_directory
from ..providers.chat_completions import (
    ChatCompletionsClient,
    ChatCompletionsConfig,
    InferenceResult,
)
from .config import load_prompt_set


@dataclass
class SmokeResult:
    """One prompt result plus a coarse status for the smoke summary."""

    result: InferenceResult
    prompt: str
    status: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a flat dict, including the smoke status."""
        data = self.result.to_dict()
        data["prompt"] = self.prompt
        data["status"] = self.status
        return data


def _default_prompts() -> tuple[str, list[str]]:
    return "You are a helpful assistant.", ["What is the capital of France?"]


def _load_prompts(
    prompt_set_path: str | None, prompt: str | None, system: str | None
) -> tuple[str, list[str]]:
    """Resolve system message and prompt list from optional prompt set or CLI args."""
    if prompt_set_path:
        prompt_set = load_prompt_set(prompt_set_path)
        return prompt_set.system or "", prompt_set.prompts

    default_system, default_prompts = _default_prompts()
    return system or default_system, ([prompt] if prompt else default_prompts)


def _synthetic_result(cfg: ChatCompletionsConfig) -> InferenceResult:
    """Return a deterministic result for --dry-run wiring checks."""
    return InferenceResult(
        provider=cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        response_text="Dry-run response.",
        finish_reason="stop",
        latency_ms=0.0,
        time_to_first_token_ms=0.0,
    )


def _run_one(
    client: ChatCompletionsClient, prompt: str, system: str, stream: bool, dry_run: bool
) -> InferenceResult:
    if dry_run:
        return _synthetic_result(client.config)
    return client.chat_simple(prompt, system=system, stream=stream)


def _status(result: InferenceResult, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    if result.error:
        return "error"
    return "ok"


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run_vllm_smoke(
    cfg: ChatCompletionsConfig,
    run_name: str,
    prompt_set: str | None = None,
    prompt: str | None = None,
    system: str | None = None,
    dry_run: bool = False,
) -> tuple[str, list[SmokeResult]]:
    """Send one or more chat-completion prompts and write smoke artifacts.

    Returns:
        Tuple of ``(run_dir, results)``.
    """
    system_msg, prompts = _load_prompts(prompt_set, prompt, system)
    if not prompts:
        raise ValueError("At least one prompt is required")

    run_dir = resolve_output_directory(os.path.join("runs", run_name))
    client = ChatCompletionsClient(cfg, run_name=run_name)

    results: list[SmokeResult] = []
    for p in prompts:
        result = _run_one(client, p, system_msg, cfg.stream, dry_run)
        results.append(
            SmokeResult(
                result=result,
                prompt=p,
                status=_status(result, dry_run),
            )
        )

    _write_artifacts(run_dir, results, cfg)

    errors = sum(1 for r in results if r.status == "error")
    logger.info(
        "Smoke complete: %s prompt(s), %s error(s). Artifacts in %s",
        len(results),
        errors,
        run_dir,
    )

    return str(run_dir), results


def _write_artifacts(run_dir: Path, results: list[SmokeResult], cfg: ChatCompletionsConfig) -> None:
    """Write the JSON result, JSONL benchmark event, and Markdown summary."""
    os.makedirs(run_dir, exist_ok=True)

    with open(run_dir / "smoke_vllm_result.json", "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)
        f.write("\n")

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(results, start=1):
            event = _to_benchmark_event(i, r, cfg)
            f.write(json.dumps(event, default=str) + "\n")

    with open(run_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(_build_summary(run_dir, results, cfg))


def _to_benchmark_event(idx: int, smoke: SmokeResult, cfg: ChatCompletionsConfig) -> dict[str, Any]:
    """Map a smoke result to the benchmark event JSONL schema."""
    r = smoke.result
    return {
        "request_id": idx,
        "model_id": cfg.model,
        "status": smoke.status,
        "stream": cfg.stream,
        "prompt_tokens": r.input_tokens,
        "completion_tokens": r.output_tokens,
        "latency_ms": r.latency_ms,
        "error": r.error or None,
        "workload": "smoke",
    }


def _build_summary(run_dir: Path, results: list[SmokeResult], cfg: ChatCompletionsConfig) -> str:
    total = len(results)
    ok = sum(1 for r in results if r.status == "ok")
    errors = sum(1 for r in results if r.status == "error")
    dry = sum(1 for r in results if r.status == "dry_run")
    latencies = [r.result.latency_ms for r in results if r.status == "ok"]
    ttfts = [r.result.time_to_first_token_ms for r in results if r.status == "ok"]
    prompt_tokens = sum(r.result.input_tokens for r in results)
    completion_tokens = sum(r.result.output_tokens for r in results)

    lines = [
        "# vLLM Compatibility Smoke Summary",
        "",
        f"- **Run:** `{run_dir.name}`",
        f"- **Model:** `{cfg.model}`",
        f"- **Base URL:** `{cfg.base_url}`",
        f"- **Stream:** `{cfg.stream}`",
        f"- **Timestamp:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Results",
        "",
        f"- **Total prompts:** {total}",
        f"- **OK:** {ok}",
        f"- **Errors:** {errors}",
        f"- **Dry-run:** {dry}",
        f"- **Mean latency (ms):** {_mean(latencies):.2f}",
        f"- **Mean TTFT (ms):** {_mean(ttfts):.2f}",
        f"- **Prompt tokens:** {prompt_tokens}",
        f"- **Completion tokens:** {completion_tokens}",
        "",
        "## Per-prompt details",
        "",
        "| Prompt | Status | Response text | Latency (ms) | TTFT (ms) | Error |",
        "|--------|--------|---------------|-------------:|----------:|-------|",
    ]
    for r in results:
        text = (r.result.response_text or "").replace("|", "\\|")[:60]
        error = (r.result.error or "").replace("|", "\\|")
        prompt = (r.prompt or "").replace("|", "\\|")
        lines.append(
            f"| {prompt} | {r.status} | {text} | {r.result.latency_ms:.2f} | "
            f"{r.result.time_to_first_token_ms:.2f} | {error} |"
        )
    lines.append("")
    return "\n".join(lines)
