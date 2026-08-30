# Agoge Forger

Agoge Forger is a Python/PyTorch-first post-training research platform for reproducible SFT, evaluation, checkpoints, experiment manifests, and Hugging Face model releases. It is designed for local RTX 5080 development with Transformers, TRL, PEFT, bitsandbytes, and vLLM-compatible model export.

## Repository boundaries

Agoge owns training, evaluation, checkpoints, manifests, and releases. It consumes versioned inputs rather than duplicating other repositories:

| Responsibility | Owner |
| --- | --- |
| Synthetic generation and public-dataset admission | [`rmems/synthetic-factory`](https://github.com/rmems/synthetic-factory) |
| Real engineering trajectories and SFT/DPO datasets | [`rmems/operation-prometheus`](https://github.com/rmems/operation-prometheus) |
| Custom CUDA kernels and GPU-performance investigations | [`rmems/blackwell-kernel-lab`](https://github.com/rmems/blackwell-kernel-lab) |
| Terraform, cloud jobs, costs, and provider runbooks | [`rmems/Dioscuri-Cloud`](https://github.com/rmems/Dioscuri-Cloud) |

There are no first-party CUDA, Rust, Julia, JAX, or cloud-infrastructure implementation trees in this repository.

## Quickstart

```bash
uv sync --all-groups
uv run agoge check-torch
uv run agoge train-qlora --config configs/minicpm5_canary.yaml
```

The canary is `openbmb/MiniCPM5-1B-Base`; the future measured flagship is `ibm-granite/granite-4.1-3b-base`. Pin an immutable Hub revision and complete the compatibility, split, and evaluation contracts before inspecting measured results.

To inspect an architecture before choosing LoRA targets:

```bash
uv run agoge model-metadata --model-id openbmb/MiniCPM5-1B-Base
uv run agoge inspect-lora-targets --model-id openbmb/MiniCPM5-1B-Base
```

`trust_remote_code` defaults to `false`; enable it only after an explicit architecture review.

## Artifact model

`checkpoint-*` directories are trainer recovery snapshots. The adapter saved at `adapters/<run_name>` is the LoRA output for continued PEFT work. The merged model under `merged/<run_name>` is the single final artifact to ship or evaluate as a standalone model.

Every training run records a reproducibility manifest, GPU telemetry, and artifact digests. Safetensors is the default artifact format; path validation rejects traversal before model, dataset, adapter, checkpoint, and output paths are used.

## vLLM compatibility smoke

```bash
uv run agoge serve-vllm --model merged/<run_name>
uv run agoge smoke-vllm --model merged/<run_name> --run-name smoke_<run_name>
```

See [vLLM model compatibility](docs/vllm_model_compatibility.md) for supported artifact forms.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/agoge_forger
uv run pytest tests/
```

The root Docker image is a CPU/smoke image. It installs from the locked dependency graph, runs as a non-root user, and never bakes in `HF_TOKEN`.

## License

Apache-2.0. See [LICENSE](LICENSE).
