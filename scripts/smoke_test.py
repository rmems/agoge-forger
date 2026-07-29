#!/usr/bin/env python3
"""Python smoke test runner for inference checks.

Produces the following artifacts in --output-dir:
  manifest.json      Experiment config snapshot
  provider.json      Runtime / provider environment info
  usage_before.json  Token-usage snapshot before the run
  usage_after.json   Token-usage snapshot after the run
  usage_delta.json   Difference (after - before)
  results.jsonl      Per-request results (one JSON object per line)
  summary.md         Human-readable summary

Artifact creation requires openat/dirfd (``os.O_DIRECTORY``), i.e. POSIX.
Platforms without that API (typical Windows CPython) refuse writes fail-closed
so path-jail guarantees are not silently weakened by pathname reopen races.
Use WSL or another POSIX environment for smoke runs that need artifacts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _supports_dirfd() -> bool:
    """True when openat + no-follow directory opens are available (POSIX).

    Requires both ``O_DIRECTORY`` and ``O_NOFOLLOW`` so nested path components
    cannot be followed if replaced by a symlink after validation.
    """
    return hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python smoke test for inference checks")
    p.add_argument("--model", required=True, help="Hugging Face model ID")
    p.add_argument("--workload", default="inference", choices=["inference", "eval", "inspect"])
    p.add_argument("--max-requests", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--stream", action="store_true", default=False)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument(
        "--output-dir",
        default=("smoke_output" if _supports_dirfd() else "."),
        help=(
            "Output directory under cwd (default: smoke_output). "
            "Requires openat/dirfd (POSIX); non-POSIX platforms exit before writing."
        ),
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow Hugging Face to execute custom Python from the model repo",
    )
    args = p.parse_args()
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if args.max_requests < 1:
        p.error("--max-requests must be >= 1")
    return args


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


def _run_inference_request(idx: int, model: str, stream: bool, dry_run: bool, trust_remote_code: bool) -> dict[str, Any]:
    if dry_run:
        return _dry_run_request(idx, model, stream)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        pt_model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=trust_remote_code, device_map="cpu"
        )
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


# Artifact basenames only — never derived from user input (breaks path-taint to open/json.dump).
_ARTIFACT_MANIFEST = "manifest.json"
_ARTIFACT_PROVIDER = "provider.json"
_ARTIFACT_USAGE_BEFORE = "usage_before.json"
_ARTIFACT_USAGE_AFTER = "usage_after.json"
_ARTIFACT_USAGE_DELTA = "usage_delta.json"
_ARTIFACT_RESULTS = "results.jsonl"
_ARTIFACT_SUMMARY = "summary.md"


def _safe_artifact_path(out_dir: Path, name: str) -> Path:
    """Join a fixed basename under a pre-sanitized output directory."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"Artifact name must be a plain basename: {name!r}")
    return out_dir / name


def _dir_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _mkdir_nofollow_under(dir_fd: int, name: str) -> int:
    """Create ``name`` under ``dir_fd`` if needed and open it without following symlinks."""
    try:
        os.mkdir(name, 0o755, dir_fd=dir_fd)
    except FileExistsError:
        pass
    return os.open(name, _dir_open_flags(), dir_fd=dir_fd)


def _open_cwd_dir_fd() -> int:
    """Open the process working directory by descriptor, not by absolute path.

    Using ``"."`` anchors the openat walk to the process cwd. Reopening
    ``Path.cwd().resolve()`` by pathname would follow a replacement if another
    process renamed the old cwd path and substituted a symlink between resolve
    and open.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    # Do not set O_NOFOLLOW on ".": POSIX cwd open is the process's own
    # working directory; O_NOFOLLOW on "." is not portable and not required.
    return os.open(".", flags)


def _open_path_under_cwd(rel_parts: tuple[str, ...], *, create: bool) -> int:
    """Open a directory under cwd via openat, never following symlink components."""
    fd = _open_cwd_dir_fd()
    try:
        for part in rel_parts:
            if part in ("", ".", ".."):
                raise ValueError(f"Invalid path segment in output directory: {part!r}")
            if create:
                next_fd = _mkdir_nofollow_under(fd, part)
            else:
                next_fd = os.open(part, _dir_open_flags(), dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def _require_under_cwd(path: Path, cwd: Path) -> Path:
    """Return ``path`` if it is under ``cwd``; otherwise raise ValueError."""
    try:
        path.relative_to(cwd)
    except ValueError as exc:
        raise ValueError(
            f"Output directory must be under the current working directory ({cwd}): {path}"
        ) from exc
    return path


def _relative_parts_under_cwd(raw: str) -> tuple[Path, tuple[str, ...]]:
    """Resolve ``raw`` under cwd without creating anything; return (cwd, parts)."""
    if not raw or not str(raw).strip():
        raise ValueError("Output directory must not be empty")

    candidate = Path(raw).expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"Path must not contain '..': {candidate}")

    cwd = Path.cwd().resolve()
    resolved = _require_under_cwd(candidate.resolve(), cwd)
    parts = tuple(p for p in resolved.relative_to(cwd).parts if p not in ("", "."))
    return cwd, parts


def _jailed_output_dir(raw: str) -> Path:
    """Sanitize ``--output-dir`` and jail it under the process cwd.

    Validate containment **before** creating anything on disk, then create
    each path component with openat + ``O_NOFOLLOW`` so intermediate symlink
    components cannot redirect the jail outside cwd.

    Platforms without dirfd/openat cannot safely create or write under nested
    directories (pathname reopen races). Fail closed: only ``--output-dir .``
    is accepted there. Default is ``smoke_output`` on POSIX and ``.`` without
    openat.
    """
    cwd, parts = _relative_parts_under_cwd(raw)
    if not parts:
        return cwd

    if not _supports_dirfd():
        raise ValueError(
            "Non-cwd --output-dir requires openat/dirfd; "
            "use --output-dir . on this platform"
        )

    dir_fd = _open_path_under_cwd(parts, create=True)
    os.close(dir_fd)
    return _require_under_cwd(cwd.joinpath(*parts).resolve(), cwd)


def _write_via_fd(fd: int, text: str) -> None:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_text_nofollow_fallback(path: Path, text: str) -> None:
    """Fail closed without openat/dirfd (no safe pathname-only write).

    Opening a temp file under cwd by absolute path races if another process
    replaces the cwd directory with a symlink between containment checks and
    ``os.open``. Without a directory handle there is no portable fix, so refuse
    rather than reintroduce an escape hatch.
    """
    del path, text
    raise ValueError(
        "Artifact writes require openat/dirfd; "
        "this platform cannot safely write smoke-test artifacts"
    )


def _unlink_at(dir_fd: int, name: str) -> None:
    """Best-effort unlink of ``name`` relative to ``dir_fd``."""
    try:
        os.unlink(name, dir_fd=dir_fd)
    except OSError:
        pass


def _write_exclusive_temp_at(dir_fd: int, tmp_name: str, text: str) -> None:
    """Create ``tmp_name`` under ``dir_fd`` with O_EXCL and write ``text``."""
    fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=dir_fd)
    try:
        _write_via_fd(fd, text)
    except Exception:
        _unlink_at(dir_fd, tmp_name)
        raise


def _replace_temp_at(dir_fd: int, tmp_name: str, final_name: str) -> None:
    """Rename temp onto final basename under ``dir_fd``; unlink temp on failure."""
    try:
        os.replace(tmp_name, final_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except Exception:
        _unlink_at(dir_fd, tmp_name)
        raise


def _write_text_nofollow_dirfd(path: Path, rel_parts: tuple[str, ...], text: str) -> None:
    """Write via exclusive temp + renameat under an openat parent directory."""
    dir_fd = _open_path_under_cwd(rel_parts, create=False) if rel_parts else _open_cwd_dir_fd()
    tmp_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        _write_exclusive_temp_at(dir_fd, tmp_name, text)
        _replace_temp_at(dir_fd, tmp_name, path.name)
    finally:
        os.close(dir_fd)


def _write_text_nofollow(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` without following symlinks or truncating hard links."""
    cwd = Path.cwd().resolve()
    rel = _require_under_cwd(path.parent.resolve(), cwd).relative_to(cwd)
    if not _supports_dirfd():
        _write_text_nofollow_fallback(path, text)
        return
    parts = tuple(p for p in rel.parts if p not in ("", "."))
    _write_text_nofollow_dirfd(path, parts, text)


def _write_json_under(out_dir: Path, name: str, data: Any) -> None:
    path = _safe_artifact_path(out_dir, name)
    _write_text_nofollow(path, json.dumps(data, indent=2) + "\n")


def _write_results_jsonl(out_dir: Path, results: list[dict[str, Any]]) -> None:
    path = _safe_artifact_path(out_dir, _ARTIFACT_RESULTS)
    _write_text_nofollow(path, "".join(json.dumps(r) + "\n" for r in results))


def _result_status_counts(results: list[dict[str, Any]]) -> tuple[int, int, int]:
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = sum(1 for r in results if r["status"] == "error")
    dry = sum(1 for r in results if r["status"] == "dry_run")
    return ok, errors, dry


def _avg_ok_latency_ms(results: list[dict[str, Any]], ok_count: int) -> float:
    if not ok_count:
        return 0.0
    total = sum(r["latency_ms"] for r in results if r["status"] == "ok")
    return round(total / ok_count, 2)


def _format_summary_md(
    results: list[dict[str, Any]], delta: dict[str, Any], *, dry_run: bool, model: str
) -> str:
    ok, errors, dry = _result_status_counts(results)
    lines = [
        "# Python Smoke Test Summary",
        "",
        f"- **Model**: `{model}`",
        f"- **Mode**: {'dry-run' if dry_run else 'live'}",
        f"- **Total requests**: {len(results)}",
        f"- **OK**: {ok} | **Errors**: {errors} | **Dry-run**: {dry}",
        f"- **Prompt tokens**: {sum(r['prompt_tokens'] for r in results)}",
        f"- **Completion tokens**: {sum(r['completion_tokens'] for r in results)}",
        f"- **Avg latency**: {_avg_ok_latency_ms(results, ok)} ms",
        "",
        "## Usage Delta",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Prompt tokens | {delta.get('tokens_prompt', 0)} |",
        f"| Completion tokens | {delta.get('tokens_completion', 0)} |",
        f"| Total tokens | {delta.get('tokens_total', 0)} |",
        f"| Requests | {delta.get('requests', 0)} |",
        "",
        "Generated by `scripts/smoke_test.py`  ",
        "Agent: Kilo agent OpenCode Go/MiniMax-M3",
    ]
    return "\n".join(lines) + "\n"


def _write_summary_under(
    out_dir: Path, results: list[dict[str, Any]], delta: dict[str, Any], *, dry_run: bool, model: str
) -> None:
    path = _safe_artifact_path(out_dir, _ARTIFACT_SUMMARY)
    _write_text_nofollow(path, _format_summary_md(results, delta, dry_run=dry_run, model=model))


def _dispatch_request(
    idx: int,
    *,
    model: str,
    stream: bool,
    dry_run: bool,
    trust_remote_code: bool,
    workload: str,
) -> dict[str, Any]:
    result = _run_inference_request(idx, model, stream, dry_run, trust_remote_code)
    if workload != "inference":
        result["workload"] = workload
    return result


def _run_workload(args: argparse.Namespace) -> list[dict[str, Any]]:
    def _one(i: int) -> dict[str, Any]:
        return _dispatch_request(
            i,
            model=args.model,
            stream=args.stream,
            dry_run=args.dry_run,
            trust_remote_code=args.trust_remote_code,
            workload=args.workload,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_one, i) for i in range(1, args.max_requests + 1)]
        return [f.result() for f in futures]


def _write_usage_after(out: Path, model: str, results: list[dict[str, Any]]) -> dict[str, Any]:
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
    _write_json_under(out, _ARTIFACT_USAGE_AFTER, usage_after)
    return usage_after


def main() -> None:
    args = _parse_args()
    if not _supports_dirfd():
        print(
            "error: smoke_test artifact writes require openat/dirfd (os.O_DIRECTORY). "
            "This platform cannot safely write jailed artifacts; use a POSIX environment "
            "(e.g. Linux or WSL).",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.trust_remote_code:
        print("WARNING: trust_remote_code enabled; model repos may execute arbitrary Python.")

    # Sanitize + jail CLI --output-dir; all writes use fixed basenames under this root.
    out = _jailed_output_dir(args.output_dir)
    model = args.model
    dry_run = args.dry_run

    _write_json_under(
        out,
        _ARTIFACT_MANIFEST,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git": _git_info(),
            "config": {
                "model_id": model,
                "workload": args.workload,
                "max_requests": args.max_requests,
                "concurrency": args.concurrency,
                "stream": args.stream,
                "dry_run": dry_run,
            },
        },
    )
    _write_json_under(out, _ARTIFACT_PROVIDER, _provider_info(model))

    usage_before = _usage_snapshot("before", model)
    _write_json_under(out, _ARTIFACT_USAGE_BEFORE, usage_before)

    results = _run_workload(args)
    _write_results_jsonl(out, results)

    usage_after = _write_usage_after(out, model, results)
    delta = _compute_delta(usage_before, usage_after)
    _write_json_under(out, _ARTIFACT_USAGE_DELTA, delta)
    _write_summary_under(out, results, delta, dry_run=dry_run, model=model)

    ok = sum(1 for r in results if r["status"] in ("ok", "dry_run"))
    print(f"Completed {len(results)} requests ({ok} ok). Artifacts in {out}/")

    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
