# Polyglot Boundaries

This document clarifies which parts of `agoge-forger` are owned by Python, Rust, and Julia, and how they interact through stable file contracts.

## Design Principles

1. **Python-first** — Training, evaluation, and model management are Python
2. **Rust is optional** — Used for performance-critical tooling (JSONL validation, workload generation, inference benchmarks)
3. **Julia is optional** — Used for scientific ML experiments and statistical analysis
4. **File contracts are the interface** — All inter-language communication goes through well-defined file schemas, not FFI
5. **No runtime dependency across language boundaries** — Each language reads/writes files independently

## Ownership Matrix

| Component                        | Python | Rust   | Julia  |
|----------------------------------|--------|--------|--------|
| Training (PyTorch, TRL, PEFT)   | **Own**| Read   | —      |
| Inference provider client        | **Own**| Read   | —      |
| Experiment config (YAML/Pydantic)| **Own**| Read   | Read   |
| Dataset JSONL validation         | Write  | **Own**| —      |
| Run manifest                     | **Own**| Read   | Read   |
| Artifact index                   | **Own**| Read   | Read   |
| Benchmark results (JSONL)        | **Own**| Read   | Read   |
| Metrics aggregation              | Write  | Read   | **Own**|
| Scientific ML (Flux, Lux, SciML) | —      | —      | **Own**|
| Model compatibility checks       | **Own**| Read   | —      |
| Export validation                | **Own**| Read   | —      |
| Workload generation              | Read   | **Own**| —      |

## File Contract Summary

| Contract                    | Writer | Readers         | Spec                                    |
|-----------------------------|--------|-----------------|-----------------------------------------|
| Dataset JSONL               | Python | Rust            | [dataset_jsonl.md](dataset_jsonl.md)    |
| Run Manifest                | Python | Rust, Julia     | [run_manifest.md](run_manifest.md)      |
| Benchmark Event JSONL       | Python | Rust, Julia     | [benchmark_event_jsonl.md](benchmark_event_jsonl.md) |
| Artifact Index              | Python | Rust, Julia     | [artifact_index.md](artifact_index.md)  |

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ Python (agoge-forger)                                   │
│                                                         │
│  configs/*.yaml ──► ExperimentConfig                    │
│       │                              │                  │
│       ▼                              ▼                  │
│  datasets/*.jsonl ──► Training ──► adapters/<run>/      │
│       │                              │                  │
│       │                              ├─ adapter weights│
│       │                              └─ artifact_index │
│       │                                                 │
│       └──► Inference ──► smoke_output/ (default)        │
│                                ├─ manifest.json         │
│                                ├─ provider.json         │
│                                ├─ usage_*.json          │
│                                ├─ results.jsonl         │
│                                └─ summary.md            │
│       └──► Eval ──► runs/<run>/smoke_eval.json          │
└─────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Rust tools   │    │ Rust tools   │    │ Julia ML     │
│              │    │              │    │              │
│ agoge-jsonl  │    │ (future)    │    │ Flux/Lux/    │
│ validates    │    │ benchgen     │    │ SciML/MLJ    │
│ JSONL rows   │    │ generates    │    │ reads        │
│              │    │ workloads   │    │ results,     │
│ reads        │    │ reads       │    │ manifest     │
│ datasets     │    │ manifest,   │    │              │
│              │    │ artifacts   │    │ writes       │
│              │    │             │    │ metrics,     │
│              │    │             │    │ predictions  │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Julia Output Conventions

The Julia smoke-test workflow (`.github/workflows/julia_smoke_test.yml`) writes to `julia_output/`:

| File                  | Format     | Description                              |
|-----------------------|------------|------------------------------------------|
| `manifest.json`       | JSON       | Timestamp, git info, config             |
| `provider.json`       | JSON       | Python version, platform, CI context     |
| `usage_before.json`   | JSON       | Pre-run token/request snapshot          |
| `usage_after.json`    | JSON       | Post-run token/request snapshot         |
| `usage_delta.json`    | JSON       | Delta (tokens, requests consumed)        |
| `results.jsonl`       | JSONL      | Per-check status line (`{"status": "ok"}`) |
| `summary.md`          | Markdown   | Human-readable smoke test summary        |

> **Note:** `metrics.json`, `predictions.csv`, and `report.md` are planned outputs for future Julia ML integration scripts, not produced by the current smoke-test workflow.

## Rust Output Conventions

The Rust smoke-test workflow (`.github/workflows/rust_smoke_test.yml`) writes to `rust_output/`:

| File                  | Format     | Description                              |
|-----------------------|------------|------------------------------------------|
| `manifest.json`       | JSON       | Timestamp, git info, config             |
| `provider.json`       | JSON       | Rust toolchain and CI context            |
| `usage_before.json`   | JSON       | Pre-run snapshot                         |
| `usage_after.json`    | JSON       | Post-run snapshot                        |
| `usage_delta.json`    | JSON       | Delta summary                            |
| `results.jsonl`       | JSONL      | Per-command status lines                 |
| `summary.md`          | Markdown   | Human-readable smoke test summary        |

The `agoge-cli validate` command prints JSONL validation reports to **stdout** (not to `runs/`).

> **Note:** `workload.jsonl` is a planned output for a future `agoge-benchgen` tool. The Rust workspace currently only contains `agoge-cli`, `agoge-jsonl`, and `agoge-gguf`.

## Optional Dependency Groups

```toml
[project.optional-dependencies]
jax = ["jax[cuda13]", "flax", "optax", "orbax-checkpoint"]
dev = ["pytest", "ruff", "mypy", "pre-commit"]
```

Rust and Julia are not declared as Python dependencies. They are standalone toolchains invoked via their respective runtimes:

- **Rust:** `cd rust-tools && cargo run -p agoge-cli -- validate <file.jsonl>`
- **Julia:** `julia --project=<project_dir> -e '...'` (see `.github/workflows/julia_smoke_test.yml`; no `julia/scripts` tree in this repo yet)

## Compatibility Guarantees

- All file schemas are **backward-compatible** — new fields may be added, but existing fields will not be removed or renamed without a major version bump
- Optional fields are documented as such and must be handled gracefully by readers
- JSON files use `indent=2` formatting for human readability
- All text files are UTF-8 encoded
- File paths in JSON use forward slashes
