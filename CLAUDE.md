@AGENTS.md

# Claude Code notes

`AGENTS.md` (imported above) is the canonical contributor guidance. This file only
adds what a Claude Code session needs on top of it.

## Commands

The gate CI runs (`.github/workflows/pr_quality_gate.yml`), all from the repo root:

```bash
uv sync --python 3.14 --frozen --extra dev   # == make setup
uv lock --check --python 3.14                # lockfile must match pyproject.toml
uv run ruff check .
uv run ruff format --check .
uv run mypy src/agoge_forger
CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q
```

- **Hide the GPU for the full suite.** `tests/test_consumer_contract.py` refuses to
  run when CUDA is visible (`consumer_contract.yml` sets `CUDA_VISIBLE_DEVICES: "-1"`;
  `pr_quality_gate.yml` does not). On this RTX 5080 host, a bare
  `uv run pytest tests/` fails those 5 tests and passes the rest.
- Makefile targets: `setup`, `check` / `check-torch` (prints torch/CUDA/VRAM),
  `test`, `lint` (ruff check + mypy; not format), `train-smoke` (QLoRA on
  `configs/minicpm5_canary.yaml`, needs the GPU and model weights), `eval-smoke`
  (needs `adapters/minicpm5_canary` from `train-smoke`).
- `uv run agoge --help` lists every CLI command (`src/agoge_forger/cli.py`).
- **Run the gate on Linux (WSL), not native Windows.** On native Windows the TRL
  import fails with a cp1252 decode error unless `PYTHONUTF8=1` is set; even then,
  mypy reports `os.O_DIRECTORY` (pass `--platform linux` to match CI) and about 300
  tests fail on POSIX-only filesystem behavior. `make` is not installed on the
  Windows side.

## Architecture map

- `src/agoge_forger/`: the only package. Entry point is `agoge = agoge_forger.cli:app`
  (Typer); the commands are in `_cli_*.py`.
  - `train/`: LoRA/QLoRA trainer (TRL `SFTConfig`), preflight, checkpoints, resume
    state, fault harness.
  - `eval/`: smoke eval, G0 and held-out base-vs-SFT generation, scoring, comparison,
    eval bundles and their schemas.
  - `export/`: adapter merge (`merge_adapter.py`) and GGUF notes.
  - `release/`: offline reproducibility-bundle verification (`agoge verify-bundle`).
  - `models/`, `artifacts/`, `telemetry/`, `providers/`, `backends/`: model loading and
    inspection, safetensors I/O, profiler markers and windows, the chat-completions
    client, and the torch environment check.
  - `split_*`, `mixture_*`, `consumer_contract.py`: frozen splits, token-budget
    mixtures, and the dataset consumer contract.
  - `run_status.py`, `_run_status_*.py`, `cleanup_run.py`: run readiness and checkpoint
    cleanup.
- `configs/`: shipped trainer YAML (`granite_4_1_flagship.yaml`, `minicpm5_canary.yaml`,
  plus a `.example`). These set `output_dir: "adapters"`.
- `datasets/`: `README.md` and `samples/tiny_sft.jsonl` only. Real data lives outside
  the repo (see AGENTS.md).
- `tests/`: pytest suite. Fixtures are in `tests/fixtures/`, plus `*_cases.py` and
  `*_fixtures.py`.
- `docs/`: design and contract notes (frozen split/eval contracts, completion-only
  training, CUDA profiling windows, RTX 5080 fit).
- `cuda/`: stub `README.md` only. This is the CUDA boundary.

## Invariants

- **The CUDA boundary.** Training, eval, merge, and export are in scope here. Custom
  `.cu`/CUTLASS kernels, engine-CUDA measurement, and host GPU scheduling belong in
  `rmems/blackwell-kernel-lab`; see `cuda/README.md`. Never add files under `cuda/`
  besides that README. In-repo GPU work stays at the torch level: the opt-in
  torch.profiler window and the correlation markers in `telemetry/`.
- **Don't edit generated output.** `adapters/`, `runs/`, `merged/`, and `gguf/` are
  gitignored run outputs (only `.gitkeep` is tracked). Don't commit their contents.
- **`reports/` is committed evidence.** The directory is gitignored, but measured
  eval/split bundles in it were force-added (about 70 tracked files). Don't rewrite or
  regenerate them by hand; they back the G0/G1 claims and are pinned by digest.
- Don't touch `.venv/`, `uv.lock` (except through `uv lock`), or the gitignored
  `.claude/` directory as part of a change.
- Analyzer configs (`.codacy.yml`, `qodana.yaml`, `.qlty/`, `.codescene/`,
  `.deepsource.toml`): follow AGENTS.md and don't widen excludes without approval.
