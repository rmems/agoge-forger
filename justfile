# Short recipes for the local agoge workflow (#205). Every recipe wraps an existing
# `uv run ...` command; nothing here adds CLI behavior. Run `just` to list recipes.
#
# Extra arguments pass straight through, quoting intact:
#   just test -k split -q
#   just train configs/granite_4_1_flagship.yaml --run-id run_001
#   just train config=configs/granite_4_1_flagship.yaml   (a leading `config=` is accepted)
#   just agoge run-status --help

set shell := ["bash", "-cu"]
set positional-arguments

# Config used by `train-smoke`; override with `just smoke_config=<path> train-smoke`.
smoke_config := "configs/minicpm5_canary.yaml"
# Adapter used by `eval-smoke`; override with `just smoke_adapter=<dir> eval-smoke`.
smoke_adapter := "adapters/minicpm5_canary"

# List every recipe
default:
    @just --list

# --- Environment and quality gate (no GPU needed) ---------------------------

# Install the locked dev environment (Python 3.14 via uv)
setup:
    uv sync --python 3.14 --frozen --extra dev

# Print torch, CUDA, and VRAM info (alias of check-torch)
check: check-torch

# Print torch, CUDA, and VRAM info
check-torch:
    uv run agoge check-torch

# Run pytest on tests/ (on a GPU host use `CUDA_VISIBLE_DEVICES= just test`)
test *args:
    uv run pytest tests/ "$@"

# Ruff check and mypy, as the Makefile `lint` target did
lint:
    uv run ruff check .
    uv run mypy src/agoge_forger

# Apply ruff formatting in place
fmt:
    uv run ruff format .

# Check ruff formatting without writing (what CI runs)
fmt-check:
    uv run ruff format --check .

# Full PR quality gate, in CI order (.github/workflows/pr_quality_gate.yml)
ci:
    uv lock --check --python 3.14
    uv sync --python 3.14 --frozen --extra dev
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src/agoge_forger
    # Hide the GPU: tests/test_consumer_contract.py refuses to run when CUDA is
    # visible, so a bare run fails those 5 tests on the RTX 5080 host (see CLAUDE.md).
    CUDA_VISIBLE_DEVICES= uv run pytest tests/

# --- CLI pass-through ---------------------------------------------------------

# Run any agoge command: `just agoge <command> [options]`
agoge *args:
    uv run agoge "$@"

# Show CLI help, or one command's help: `just help train-qlora`
help *command:
    uv run agoge "$@" --help

# Freeze an immutable three-way split: `just freeze --source ... --output-dir ...`
freeze *args:
    uv run agoge freeze-split "$@"

# --- GPU recipes: need CUDA, model weights, and (for eval) a trained adapter ----

# [CUDA] QLoRA canary training on smoke_config
train-smoke:
    uv run agoge train-qlora --config {{ quote(smoke_config) }}

# [CUDA] Smoke eval of the canary adapter (run train-smoke first)
eval-smoke:
    uv run agoge smoke-eval --adapter-path {{ quote(smoke_adapter) }}

# [CUDA] QLoRA training: `just train <config.yaml> [--run-id ...] [--profile-window ...]`
train config *args:
    uv run agoge train-qlora --config "${1#config=}" "${@:2}"

# [CUDA] LoRA (non-quantized) training: `just train-lora <config.yaml> [options]`
train-lora config *args:
    uv run agoge train-lora --config "${1#config=}" "${@:2}"

# [CUDA] Paired base-vs-SFT held-out eval: `just eval --split-manifest ... --sft-artifact ...`
eval *args:
    uv run agoge held-out-eval "$@"

# [CUDA] Base-only (G0) held-out eval before SFT: `just eval-g0 --split-manifest ...`
eval-g0 *args:
    uv run agoge g0-held-out-eval "$@"
