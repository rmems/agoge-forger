# Benchmark Event JSONL Schema

Benchmark and smoke-test results are logged as newline-delimited JSON (JSONL). The schema depends on the producer:

- **Python inference smoke test** (`scripts/smoke_test.py`): one object per inference request with token and latency fields (documented below).

## File Location

On POSIX hosts, `scripts/smoke_test.py` writes `results.jsonl` and the companion
files below to its `--output-dir` (default `smoke_output/`, gitignored). Platforms
without `openat`/dirfd support (typical Windows CPython) fail closed and exit before
writing anything; run the script under WSL or another POSIX environment. Training
runs do not produce this file; what lands under `runs/<run_name>/` is specified in
`docs/contracts/run_manifest.md`.

## Schema

Each line is a JSON object:

```json
{
  "request_id": 1,
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "ok",
  "stream": true,
  "prompt_tokens": 24,
  "completion_tokens": 128,
  "latency_ms": 342.5,
  "error": null,
  "workload": "eval"
}
```

## Fields

| Field                | Type        | Required | Description                                     |
|----------------------|-------------|----------|-------------------------------------------------|
| `request_id`         | int         | Yes      | Sequential request number (1-indexed)           |
| `model_id`           | str         | Yes      | Model identifier used for the request           |
| `status`             | str         | Yes      | One of: `"ok"`, `"error"`, `"dry_run"`          |
| `stream`             | bool        | Yes      | Whether streaming was enabled                   |
| `prompt_tokens`      | int         | Yes      | Number of tokens in the prompt                  |
| `completion_tokens`  | int         | Yes      | Number of tokens in the completion              |
| `latency_ms`         | float       | Yes      | End-to-end request latency in milliseconds      |
| `error`              | str or null | Yes      | Error message if `status` is `"error"`, else `null` |
| `workload`           | str         | No       | Present for `eval` or `inspect` workloads only  |

## Companion Files

The benchmark runner also produces:

### `manifest.json`

```json
{
  "timestamp": "2025-01-15T14:30:00Z",
  "git": { "commit": "abc1234", "branch": "main", "dirty": false },
  "config": {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "workload": "inference",
    "max_requests": 100,
    "concurrency": 1,
    "stream": true,
    "dry_run": false
  }
}
```

### `provider.json`

```json
{
  "timestamp": "2025-01-15T14:30:00Z",
  "python_version": "3.12.0",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "platform": "linux",
  "ci": false,
  "github_actions": false,
  "runner_os": "",
  "runner_arch": ""
}
```

### `usage_before.json` / `usage_after.json`

```json
{
  "label": "before",
  "timestamp": "2025-01-15T14:30:00Z",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens_prompt": 0,
  "tokens_completion": 0,
  "tokens_total": 0,
  "requests": 0
}
```

### `usage_delta.json`

```json
{
  "label": "delta",
  "timestamp": "2025-01-15T14:30:00Z",
  "tokens_prompt": 2400,
  "tokens_completion": 12800,
  "tokens_total": 15200,
  "requests": 100
}
```

### `summary.md`

Human-readable markdown summary of the benchmark run.

## Owner

Python writes and consumes benchmark events, usage, and summaries.
