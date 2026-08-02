# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup                     # uv pip install -e ".[dev]" (installs the `agoge` console script)
make test                      # pytest tests/
make lint                      # ruff check . && mypy src/
make check                     # check-torch + check-jax + cargo check
make train-smoke               # agoge train-qlora --config configs/smoke_test.yaml  (needs CUDA)
make eval-smoke                # agoge smoke-eval --adapter-path adapters/smoke_test_run
make rust-check                # cd rust-tools && cargo check
make cuda-build / cuda-test    # optional C++/CUDA dummy extension
make tf-validate PROVIDER=aws  # terraform validate for infra/terraform/environments/<provider>
```

Single test / focused runs (CI uses the venv binaries directly, and so should you if `agoge` is not on PATH):

```bash
.venv/bin/pytest tests/test_checkpoints.py::test_name -q
.venv/bin/pytest tests/ -q -k checkpoint
```

CI quality gate (`.github/workflows/pr_quality_gate.yml`) runs exactly: `ruff check .`, `ruff format --check .`, `mypy src/agoge_forger`, `pytest tests/`. Note the Makefile's `lint` target uses `mypy src/` — CI's narrower `src/agoge_forger` is the authoritative check. Rust CI additionally runs `cargo clippy -- -D warnings` from `rust-tools/`.

**Check your local tool versions against `pyproject.toml` before trusting a green run.** CI builds a fresh venv, so it gets exactly the pinned `ruff` (and the newest `transformers`/`trl`, since `uv pip install -e ".[dev]"` ignores `uv.lock`). A long-lived `.venv` drifts: a stale ruff has passed locally on code that the pinned ruff then rejected in CI. `.venv/bin/ruff --version` should match the `ruff==` pin.

The whole Python test suite is CPU-only, network-free, and runs in a few seconds (~110 tests). It never downloads models: tests use hand-rolled `Dummy*` stand-ins, `tmp_path` fixtures, and — in `test_trainer_trl_api.py` — a model built from a bare `LlamaConfig` plus an in-memory word-level tokenizer. Keep it that way; anything requiring a GPU or the Hub belongs behind the smoke-test workflows, not `tests/`.

`tests/test_trainer_trl_api.py` deliberately constructs a **real** `trl.SFTTrainer` rather than mocking it. TRL/Transformers churn their kwargs between releases (`max_seq_length`→`SFTConfig.max_length`, `tokenizer`→`processing_class`, `save_safetensors` deleted outright), and that churn once broke every training run while the suite stayed green. Mocks would defeat the point — and `MagicMock` does not work anyway, since TRL type-checks `processing_class`.

## Architecture

A local-first LLM fine-tuning forge (QLoRA/LoRA on a 16GB RTX 5080 by default). Python owns everything real; Rust/Julia/CUDA/Terraform are optional side toolchains.

### CLI is the only entry point

`src/agoge_forger/cli.py` (Typer, exposed as `agoge`) is the single surface. Everything under `scripts/*.py` is either a three-line wrapper that calls `agoge_forger.cli:app` with fixed args, or a standalone CI harness (`scripts/smoke_test.py`). When adding functionality: put the logic in a subpackage, wire it into `cli.py`, and only add a `scripts/` shim if a workflow needs a bare `python` invocation.

`cli.py` also owns the **security boundary**: it resolves and validates every user-supplied path (`resolve_existing_path` / `resolve_output_directory`) and enforces the safetensors policy *before* calling into library code. Library functions re-check the same invariants (see `merge_adapter`), so direct library callers stay safe — this duplication is deliberate defense in depth, not dead code.

### Training pipeline

`train/qlora.py` and `train/lora.py` are thin config mutators over the shared `train/trainer.py:run_training`, which is the one place the pipeline lives:

1. `preflight.check_cuda_available(required=True)` — training hard-fails without a GPU.
2. `preflight.get_gpu_report` / `estimate_training_risk` / `warn_on_disk_pressure` — warnings only, based on VRAM and free disk under `output_dir`.
3. `models/load.py:load_base_model` — tokenizer + model, BitsAndBytes 4-bit config, pad-token fallback to EOS.
4. `_prepare_peft_model` → `preflight.validate_lora_targets_exist` — see LoRA targets below.
5. `datasets.py:load_jsonl_dataset` → TRL `SFTTrainer`.
6. `checkpoints.resolve_resume_checkpoint` → `trainer.train(resume_from_checkpoint=...)`.
7. `_finalize_training_run` — save adapter, assert no unsafe bins, write `artifact_index.json`, write run manifest.

### Output layout (three distinct artifact kinds)

| Path | Written by | Meaning |
|---|---|---|
| `adapters/<run_name>/` | trainer save | The LoRA adapter — the thing you keep training from |
| `adapters/<run_name>/checkpoint-N/` | HF Trainer | Recovery snapshots only, capped by `save_total_limit` |
| `runs/<run_name>/manifest.json` | `manifests.py` | Reproducibility record (git state, env, GPU telemetry, metrics) |
| `merged/<run_name>/` | `export/merge_adapter.py` | Single shippable merged model |

Non-obvious: the manifest goes to `runs/<run_name>/`, **not** to `output_dir`. `runs/<run_name>/` also collects `smoke_eval.json` and raw provider responses. Every output directory gets an `artifact_index.json` (relative path, size, SHA256 per file).

### Config: flat YAML → nested Pydantic

`config.py` defines a nested `ExperimentConfig` (`quantization` / `training` / `lora` / `runtime` sub-models) but the YAML files in `configs/` are **flat**. `load_config` hand-maps every flat key into the nested model. Adding a config option therefore means editing two places: the sub-model field *and* the corresponding `data.get(...)` line in `load_config`. Forgetting the second means the YAML key is silently ignored.

Relative `dataset_path` values resolve against the **config file's directory**, not the CWD — which is why `configs/*.yaml` say `../datasets/samples/tiny_sft.jsonl`.

### Safety invariants (these are the point of the project — do not relax them)

- **safetensors-only.** `artifacts/safetensors_io.py:UNSAFE_WEIGHT_PATTERNS` blocks `pytorch_model.bin`, `adapter_model.bin`, `*.ckpt`. It deliberately does *not* match the Trainer's `*.pt` optimizer/RNG state under `checkpoint-*`. Every adapter-consuming entry point (`smoke-eval`, `merge-adapter`, `export-final-model`, resume) runs `assert_no_unsafe_weight_bins` unless `--allow-unsafe-serialization` is passed.
- **`trust_remote_code` defaults to `false`** everywhere (CLI flag, YAML key, `load_base_model` argument), and loading with it enabled logs a warning.
- **No `..` in paths.** `path_safety.py` checks the *pre-resolution* path parts, because `Path.resolve()` would consume the traversal.
- **LoRA targets must exist in the loaded model graph.** `target_modules_mode` selects the strictness: `auto_common` (intersect with the standard q/k/v/o/gate/up/down set), `explicit` (raise if a named target is absent), `discover_required` (raise if no targets were supplied). Training aborts when nothing validates. Use `agoge inspect-lora-targets` / `agoge model-metadata` for unknown architectures instead of guessing — see `docs/lora_target_discovery.md`.
- **A checkpoint is "valid"** only if it is a `checkpoint-N` directory containing `trainer_state.json` *and* passing the adapter-artifact check (`checkpoints.py`). Resume order: explicit `resume_checkpoint_path` → latest valid checkpoint when `resume_from_latest_checkpoint` → fresh run.

### Polyglot boundaries

`docs/contracts/polyglot_boundaries.md` is the governing document: languages communicate through **file contracts only — no FFI, no cross-language runtime dependency**. Python writes; Rust and Julia read. Contract specs live in `docs/contracts/` (`dataset_jsonl.md`, `run_manifest.md`, `artifact_index.md`, `benchmark_event_jsonl.md`) and are backward-compatible — add fields, never remove or rename them. Rust (`rust-tools/`: `agoge-cli`, `agoge-jsonl`, `agoge-gguf`) does JSONL *syntax* validation only; row-schema validation stays Python-owned in `datasets.py:normalize_row` (accepts `text`, `messages`, or `instruction` rows). `backends/jax_backend.py`, `cuda/`, and `infra/terraform/` are scaffolds/stubs.

## Gotchas

- `tests/test_docs_workflow.py` asserts on **exact sentences in README.md**. Rewording the checkpoint/adapter/merged-artifact paragraph breaks the test suite; update both together.
- The root `.gitignore` lists `Cargo.lock`, but `rust-tools/Cargo.lock` is tracked on purpose (`docs/dependency_policy.md`) — don't "fix" this by deleting it. Conversely, `rust-tools/**/sbom.json` and `*.cdx.json` are build artifacts and must not be committed.
- `scripts/smoke_test.py` is not importable as a package; its tests load it via `importlib` from a file path. It also fails closed on platforms without `os.O_DIRECTORY` (use WSL, not native Windows).
- Commits follow Conventional Commits (`feat:`, `fix:`, `ci:`, `chore(deps):`, `security:`, `docs:`), and work lands through PRs.
