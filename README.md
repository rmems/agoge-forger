# Agoge Forger

Agoge Forger is a Python/PyTorch-first post-training research platform for
reproducible supervised fine-tuning, evaluation, checkpoints, experiment
manifests, and Hugging Face model releases.

The repository deliberately owns one implementation path: Python with PyTorch,
Transformers, TRL, and PEFT. PyTorch may use CUDA, bitsandbytes, and vLLM, but
Agoge does not own custom GPU kernels, alternate training runtimes, synthetic
data generation, or cloud infrastructure. See
[Repository boundaries](docs/repository_boundaries.md) for the maintained
ownership contract.

## Core workflow

Agoge is being hardened around a compact path from a versioned dataset to a
release candidate:

1. validate a committed training configuration and immutable dataset revision;
2. run a PyTorch/TRL/PEFT post-training job;
3. save bounded recovery checkpoints and the final adapter;
4. evaluate the base and adapted model against a held-out split;
5. record manifests, environment details, and content hashes; and
6. merge or export a model artifact for compatibility checks and release.

This sequence is the repository contract, not a claim that every roadmap stage
is already complete. A capability is ready only when its CLI, contract, and
validation land together; the smoke commands below prove plumbing rather than
model quality.

Model and dataset licenses remain independent from Agoge's Apache-2.0 source
license. Review [Third-party licensing](THIRD_PARTY_NOTICES.md) before
redistributing an artifact.

## Quickstart

Install the locked development environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra dev
uv run agoge check-torch
```

Run the committed smoke configuration against its tiny fixture dataset:

```bash
uv run agoge train-qlora --config configs/smoke_test.yaml
```

The smoke configuration is a plumbing check, not a model-quality experiment.
Do not publish quality claims from it. Measured runs must pin the dataset and
model revisions and preserve the resulting manifest.

## Evaluate and export

Evaluate an adapter:

```bash
uv run agoge smoke-eval --adapter-path adapters/<run_name>
```

Resume from the latest valid checkpoint by setting these configuration fields:

```yaml
save_steps: 50
save_total_limit: 2
resume_from_latest_checkpoint: true
```

Export one final model artifact from a completed run:

```bash
uv run agoge export-final-model \
  --run-dir adapters/<run_name> \
  --out-dir merged/<run_name>
```

`checkpoint-*` directories are trainer recovery snapshots.

The adapter saved at `adapters/<run_name>` is the LoRA output for continued PEFT work.

The merged model under `merged/<run_name>` is the single final artifact to ship or evaluate as a standalone model.

Serve and smoke-test that artifact with ordinary vLLM:

```bash
uv run agoge serve-vllm --model merged/<run_name>
uv run agoge smoke-vllm \
  --model merged/<run_name> \
  --run-name smoke_<run_name>
```

See [vLLM model compatibility](docs/vllm_model_compatibility.md) and
[chat-completions providers](docs/chat_completions_providers.md) for details.

## Safety and reproducibility

- Remote model code is opt-in. Keep `trust_remote_code: false` unless a reviewed
  architecture gate documents why it is necessary.
- LoRA target modules must be discovered from and validated against the loaded
  model graph; do not guess module names.
- Safetensors is the required weight format.
- Output paths reject parent-directory traversal.
- Artifact indexes record file sizes and SHA-256 hashes.
- Run manifests capture source state, package versions, tokenizer metadata, and
  available GPU telemetry.
- Checkpoint retention is bounded and resume selects only valid checkpoints.

## Dependency source of truth

Human-maintained dependency declarations live in `pyproject.toml`; the committed
`uv.lock` is the reproducible resolution. Do not maintain a second hand-edited
requirements file. See [Dependency policy](docs/dependency_policy.md).

## Docker smoke image

The root Dockerfile builds the locked CPU/smoke image:

```bash
docker build -t agoge-forger:local .
docker run --rm agoge-forger:local agoge --help
docker run --rm agoge-forger:local \
  python -c "import agoge_forger; print(agoge_forger.__version__)"
```

The image installs from `uv.lock`, runs as a non-root user, and does not bake in
Hugging Face credentials. Pass credentials only at runtime when needed.

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/agoge_forger
uv run pytest tests/
uv build
```

The project is licensed under the [Apache License 2.0](LICENSE). Contributions
are accepted under that license unless explicitly stated otherwise.
