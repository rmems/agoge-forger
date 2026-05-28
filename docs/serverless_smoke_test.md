# Serverless Smoke Test Workflows

This project includes three manual GitHub Actions workflows for smoke testing across Python, Rust, and Julia toolchains.

## Available Workflows

| Workflow | File | Language |
|----------|------|----------|
| Python Smoke Test | `.github/workflows/serverless_smoke_test.yml` | Python |
| Rust Smoke Test | `.github/workflows/rust_smoke_test.yml` | Rust |
| Julia Smoke Test | `.github/workflows/julia_smoke_test.yml` | Julia |

## Launching a Workflow

1. Go to the **Actions** tab in the GitHub repository.
2. Select the workflow from the left sidebar.
3. Click **Run workflow** (top right).
4. Fill in the input parameters.
5. Click **Run workflow** to start.

All three workflows are **manual-only** (`workflow_dispatch`). They will never trigger automatically on push or PR events.

## Python: Python Smoke Test

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `model` | `HuggingFaceM4/tiny-random-LlamaForCausalLM` | Hugging Face model ID |
| `workload` | `inference` | Workload type: `inference`, `eval`, `inspect` |
| `max_requests` | `5` | Maximum inference requests |
| `concurrency` | `1` | Request concurrency |
| `stream` | `false` | Enable streaming |
| `dry_run` | `false` | Dry-run mode (no real inference) |
| `upload_artifacts` | `true` | Upload run artifacts |

### Artifacts

- `manifest.json` — Experiment config snapshot
- `provider.json` — Runtime environment info
- `usage_before.json` — Token-usage snapshot before the run
- `usage_after.json` — Token-usage snapshot after the run
- `usage_delta.json` — Usage difference (after − before)
- `results.jsonl` — Per-request results (one JSON object per line)
- `summary.md` — Human-readable summary

### Dry-Run Mode

Set `dry_run` to `true` to validate the workflow without making real inference calls. All requests will report `status: dry_run` with zero tokens and latency.

## Rust Smoke Test

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `command` | `check` | Cargo command: `check`, `test`, `clippy`, `fmt`, `build` |
| `package` | (empty) | Specific crate (leave empty for full workspace) |
| `dry_run` | `false` | Validate Cargo.toml only, no compilation |
| `upload_artifacts` | `true` | Upload run artifacts |

### Dry-Run Mode

When `dry_run` is `true`, the workflow validates the workspace structure and `Cargo.toml` without compiling. It lists workspace members and confirms the project is valid.

## Julia Smoke Test

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `command` | `check` | Julia command: `check`, `test`, `build` |
| `project_dir` | `.` | Julia project directory |
| `julia_version` | `1.10` | Julia version to install |
| `dry_run` | `false` | Validate Julia version only |
| `upload_artifacts` | `true` | Upload run artifacts |

### Dry-Run Mode

When `dry_run` is `true`, the workflow only verifies that the requested Julia version is installed and functional.

## Local Testing

Run the Python smoke test locally:

```bash
python scripts/serverless_smoke_test.py \
  --model HuggingFaceM4/tiny-random-LlamaForCausalLM \
  --workload inference \
  --max-requests 3 \
  --concurrency 1 \
  --dry-run \
  --output-dir smoke_output
```

Run the Rust smoke test locally:

```bash
cd rust-tools && cargo check
cd rust-tools && cargo test
cd rust-tools && cargo clippy -- -D warnings
```

Run the Julia smoke test locally:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

---

*Agent: Kilo agent Vultr/MiniMax-M2.7*
