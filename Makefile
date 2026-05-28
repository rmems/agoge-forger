.PHONY: setup check check-torch check-jax inspect-gka train-smoke eval-smoke test lint rust-check cuda-build cuda-test tf-validate

setup:
	uv pip install -e ".[dev]" || pip install -e ".[dev]"

check: check-torch check-jax rust-check

check-torch:
	agoge check-torch

check-jax:
	agoge check-jax

inspect-gka:
	agoge inspect-model --model-id amazon/GKA-primed-HQwen3-8B-Reasoner

train-smoke:
	agoge train-qlora --config configs/smoke_test.yaml

eval-smoke:
	agoge smoke-eval --adapter-path adapters/smoke_test_run

test:
	pytest tests/

lint:
	ruff check .
	mypy src/

rust-check:
	cd rust-tools && cargo check

cuda-build:
	cd cuda && python setup.py build_ext --inplace

cuda-test:
	python -c "from agoge_forger.cuda.extension_loader import load_dummy_extension; load_dummy_extension()"

tf-validate:
	cd infra/terraform/environments/$(PROVIDER) && terraform init -backend=false && terraform validate
