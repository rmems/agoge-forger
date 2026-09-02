.PHONY: setup check check-torch train-smoke eval-smoke test lint

setup:
	uv pip install -e ".[dev]" || pip install -e ".[dev]"

check: check-torch

check-torch:
	agoge check-torch

train-smoke:
	agoge train-qlora --config configs/smoke_test.yaml

eval-smoke:
	agoge smoke-eval --adapter-path adapters/smoke_test_run

test:
	pytest tests/

lint:
	ruff check .
	mypy src/
