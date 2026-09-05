.PHONY: setup check check-torch train-smoke eval-smoke test lint

setup:
	uv sync --all-groups --extra dev

check: check-torch

check-torch:
	uv run agoge check-torch

train-smoke:
	uv run agoge train-qlora --config configs/minicpm5_canary.yaml

# Needs a trained adapter directory (for example adapters/<run_name> after train-smoke).
eval-smoke:
	uv run agoge smoke-eval --adapter-path adapters/minicpm5_canary

test:
	uv run pytest tests/

lint:
	uv run ruff check .
	uv run mypy src/agoge_forger
