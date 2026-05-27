#!/usr/bin/env python3
"""Serverless inference smoke test runner.

Produces the following artifacts in --output-dir:
  manifest.json      Experiment config snapshot
  provider.json      Runtime / provider environment info
  usage_before.json  Token-usage snapshot before the run
  usage_after.json   Token-usage snapshot after the run
  usage_delta.json   Difference (after - before)
  results.jsonl      Per-request results (one JSON object per line)
  summary.md         Human-readable summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serverless inference smoke test")
    p.add_argument("--model", required=True, help="Hugging Face model ID")
    p.add_argument("--workload", default="inference", choices=["inference", "eval", "inspect"])
    p.add_argument("--max-requests", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--stream", action="store_true", default=False)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--output-dir", default="smoke_output")
    return p.parse_args()


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {}


def _provider_info(model: str) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "model_id": model,
        "platform": sys.platform,
        "ci": os.getenv("CI", "false") == "true",
        "github_actions": os.getenv("GITHUB_ACTIONS", "false") == "true",
        "runner_os": os.getenv("RUNNER_OS", "local"),
        "runner_arch": os.getenv("RUNNER_ARCH", "unknown"),
    }


def _usage_snapshot(label: str, model: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "requests": 0,
    }
    if extra:
        snap.update(extra)
    return snap


def _compute_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "delta",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tokens_prompt": after.get("tokens_prompt", 0) - before.get("tokens_prompt", 0),
        "tokens_completion": after.get("tokens_completion", 0) - before.get("tokens_completion", 0),
        "tokens_total": after.get("tokens_total", 0) - before.get("tokens_total", 0),
        "requests": after.get("requests", 0) - before.get("requests", 0),
    }


def _dry_run_request(idx: int, model: str, stream: bool) -> dict[str, Any]:
    return {
        "request_id": idx,
        "model_id": model,
        "status": "dry_run",
        "stream": stream,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "error": None,
    }


def _run_inference_request(idx: int, model: str, stream: bool, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return _dry_run_request(idx, model, stream)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, device_map="cpu")
        prompt = f"Hello world request {idx}"
        inputs = tokenizer(prompt, return_tensors="pt")
        t0 = time.monotonic()
        output_ids = pt_model.generate(**inputs, max_new_tokens=32)
        elapsed = (time.monotonic() - t0) * 1000
        prompt_len = inputs["input_ids"].shape[-1]
        completion_len = output_ids.shape[-1] - prompt_len
        del pt_model
        return {
            "request_id": idx,
            "model_id": model,
            "status": "ok",
            "stream": stream,
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "latency_ms": round(elapsed, 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "request_id": idx,
            "model_id": model,
            "status": "error",
            "stream": stream,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
            "error": str(exc),
        }


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_summary(path: Path, results: list[dict[str, Any]], delta: dict[str, Any], dry_run: bool, model: str) -> None:
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = sum(1 for r in results if r["status"] == "error")
    dry = sum(1 for r in results if r["status"] == "dry_run")
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    avg_latency = (
        round(sum(r["latency_ms"] for r in results if r["status"] == "ok") / max(ok, 1), 2)
        if ok
        else 0.0
    )

    lines = [
        "# Serverless Smoke Test Summary",
        "",
        f"- **Model**: `{model}`",
        f"- **Mode**: {'dry-run' if dry_run else 'live'}",
        f"- **Total requests**: {len(results)}",
        f"- **OK**: {ok} | **Errors**: {errors} | **Dry-run**: {dry}",
        f"- **Prompt tokens**: {total_prompt}",
        f"- **Completion tokens**: {total_completion}",
        f"- **Avg latency**: {avg_latency} ms",
        "",
        "## Usage Delta",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Prompt tokens | {delta.get('tokens_prompt', 0)} |",
        f"| Completion tokens | {delta.get('tokens_completion', 0)} |",
        f"| Total tokens | {delta.get('tokens_total', 0)} |",
        f"| Requests | {delta.get('requests', 0)} |",
        "",
        "Generated by `scripts/serverless_smoke_test.py`  ",
        f"Agent: Kilo agent Vultr/MiniMax-M2.7",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = args.model
    dry_run = args.dry_run
    stream = args.stream
    max_req = args.max_requests
    workload = args.workload

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "config": {
            "model_id": model,
            "workload": workload,
            "max_requests": max_req,
            "concurrency": args.concurrency,
            "stream": stream,
            "dry_run": dry_run,
        },
    }
    _write_json(out / "manifest.json", manifest)

    _write_json(out / "provider.json", _provider_info(model))

    usage_before = _usage_snapshot("before", model)
    _write_json(out / "usage_before.json", usage_before)

    results: list[dict[str, Any]] = []
    for i in range(1, max_req + 1):
        if workload == "inference":
            result = _run_inference_request(i, model, stream, dry_run)
        elif workload == "eval":
            result = _dry_run_request(i, model, stream)
            result["workload"] = "eval"
        else:
            result = _dry_run_request(i, model, stream)
            result["workload"] = "inspect"
        results.append(result)

    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    usage_after = _usage_snapshot(
        "after",
        model,
        {
            "tokens_prompt": total_prompt,
            "tokens_completion": total_completion,
            "tokens_total": total_prompt + total_completion,
            "requests": len(results),
        },
    )
    _write_json(out / "usage_after.json", usage_after)

    delta = _compute_delta(usage_before, usage_after)
    _write_json(out / "usage_delta.json", delta)

    _write_summary(out / "summary.md", results, delta, dry_run, model)

    ok = sum(1 for r in results if r["status"] in ("ok", "dry_run"))
    print(f"Completed {len(results)} requests ({ok} ok). Artifacts in {out}/")

    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
