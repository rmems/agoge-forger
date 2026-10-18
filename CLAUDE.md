@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Project rules, the pre-PR check list, and learned preferences are in `AGENTS.md`
(imported above); do not duplicate them here.

## Commands

Run from the repo root with Python 3.14 (uv installs it). "Verified" means it ran successfully in a fresh
checkout on Linux, CPU only, no GPU.

| Command | Status |
|---|---|
| `uv sync --python 3.14 --frozen --extra dev` | verified (about 3 min cold, pulls torch/CUDA wheels) |
| `uv lock --check --python 3.14` | verified |
| `uv run ruff check .` | verified |
| `uv run ruff format --check .` | verified |
| `uv run mypy src/agoge_forger` | verified |
| `uv run pytest tests/ -q` | verified (984 passed, 8 skipped) |
| `uv run agoge check-torch` / `make check` | not run (needs a working torch/CUDA setup) |
| `uv run agoge train-qlora --config configs/minicpm5_canary.yaml` / `make train-smoke` | not run (needs the RTX 5080 and Hub downloads) |
| `make eval-smoke` | not run (needs a trained adapter under `adapters/`) |
| `uv run python scripts/smoke_test.py --model <hf-model-id>` | not run (CI `workflow_dispatch` only; `--model` is required and needs a downloaded model) |

CI: `pr_quality_gate.yml` runs the lock check, ruff, mypy, and pytest above. `config_contract_gate.yml`
loads every top-level `configs/*.yaml` and runs three test files. `consumer_contract.yml` runs
`tests/test_consumer_contract.py`. Qodana, Codacy, Qlty, CodeScene, and DeepSource are external analyzers, not local gates.

## Architecture map

- `src/agoge_forger/cli.py` + `_cli_*.py`: Typer app (`agoge`) split per command group (train, eval, data, export, release, runs, serving).
- `config.py`, `datasets.py`, `split_*.py`, `mixture_*.py`: experiment config, dataset JSONL parsing, frozen splits, and mixture composition.
- `train/`: QLoRA/LoRA trainer, checkpoints, resume state, preflight, and the checkpoint fault harness.
- `eval/`: G0/G1 held-out generation, scoring, comparison, and the evaluation/experiment contracts.
- `release/`: release verification. `export/`: adapter merge. `artifacts/`: safetensors I/O and provenance.
- `run_status.py` + `_run_status_*.py`, `cleanup_run.py`: run readiness inspection and checkpoint pruning.
- `serving/`, `providers/`, `telemetry/`, `backends/`, `models/`: vLLM serve/smoke, chat-completions provider, GPU telemetry and profile windows, torch backend, model load/inspection.
- `docs/contracts/`: schema docs for manifests, metrics, artifact index, and JSONL formats. `tests/test_docs_workflow.py` guards docs.

## Invariants and non-goals

- Packaging lives in `pyproject.toml` and `uv.lock` only; use `--frozen` and never add a requirements file.
- `trust_remote_code` stays `false` in shipped configs unless a reviewed, documented exception is needed (`AGENTS.md`). Safetensors is the default artifact format; path validation rejects traversal.
- No first-party CUDA, Rust, Julia, JAX, or cloud trees. Kernel work goes to `rmems/blackwell-kernel-lab`, cloud to `rmems/Dioscuri-Cloud`.
- Frozen split membership and pinned dataset digests must not change. Shipped configs avoid host-specific paths.
- `ruff` is pinned to `0.17.0` in `pyproject.toml` and `.pre-commit-config.yaml`; keep them in sync. Line length is 100.
- `.py` files in this repo have no SPDX header convention; do not add one.

## Gotchas

- `adapters/`, `merged/`, `runs/`, `reports/`, and `.claude/` are gitignored (some `reports/` files are tracked); keep shared agent guidance in this file or `AGENTS.md`.
- Never run publish or upload commands (PyPI, Hugging Face Hub) from an agent session.
- Edit only what the task needs; keep PRs small.
