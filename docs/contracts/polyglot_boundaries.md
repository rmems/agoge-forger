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
│       │                              ├─ manifest.json   │
│       │                              ├─ artifact_index  │
│       │                              └─ adapter weights │
│       │                                                 │
│       └──► Inference ──► runs/<run>/                    │
│                                ├─ raw/*.json            │
│                                ├─ smoke_eval.json       │
│                                └─ results.jsonl         │
└─────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Rust tools   │    │ Rust tools   │    │ Julia ML     │
│              │    │              │    │              │
│ agoge-jsonl  │    │ agoge-      │    │ Flux/Lux/    │
│ validates    │    │ benchgen    │    │ SciML/MLJ    │
│ JSONL rows   │    │ generates   │    │ reads        │
│              │    │ workloads   │    │ results,     │
│ reads        │    │ reads       │    │ manifest     │
│ datasets     │    │ manifest,   │    │              │
│              │    │ artifacts   │    │ writes       │
│              │    │             │    │ metrics,     │
│              │    │             │    │ predictions  │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Julia Output Conventions

Julia smoke scripts write to `runs/<run_name>/julia/`:

| File                      | Format | Description                       |
|---------------------------|--------|-----------------------------------|
| `metrics.json`            | JSON   | Training/evaluation metrics       |
| `predictions.csv`         | CSV    | Model predictions                 |
| `report.md`              | Markdown | Human-readable experiment report |

## Rust Output Conventions

Rust tools write to `runs/<run_name>/`:

| File                | Format | Tool          | Description                        |
|---------------------|--------|---------------|------------------------------------|
| `workload.jsonl`    | JSONL  | agoge-benchgen | Generated workload entries        |
| Validation reports  | stdout | agoge-jsonl   | JSONL validation results           |

## Optional Dependency Groups

```toml
[project.optional-dependencies]
jax = ["jax[cuda13]", "flax", "optix", "orbax-checkpoint"]
dev = ["pytest", "ruff", "mypy", "pre-commit"]
```

Rust and Julia are not declared as Python dependencies. They are standalone toolchains invoked via their respective runtimes:

- **Rust:** `cargo run --package agoge-jsonl -- <args>`
- **Julia:** `julia --project=julia julia/scripts/<script>.jl`

## Compatibility Guarantees

- All file schemas are **backward-compatible** — new fields may be added, but existing fields will not be removed or renamed without a major version bump
- Optional fields are documented as such and must be handled gracefully by readers
- JSON files use `indent=2` formatting for human readability
- All text files are UTF-8 encoded
- File paths in JSON use forward slashes
