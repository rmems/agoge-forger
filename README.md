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

## Inspect run readiness

Before resuming or exporting, ask a run directory what it is actually ready for:

```bash
agoge run-status adapters/<run_name>
agoge run-status adapters/<run_name> --format table
agoge run-status adapters/<run_name> --merged-dir merged/<custom_name>
agoge run-status adapters/<run_name> --allow-unsafe-serialization
```

The default `--format json` report, for a run with two checkpoints, a saved
adapter, and an exported merge (absolute paths shortened, some fields elided):

```json
{
  "schema_version": 1,
  "run_name": "demo_run",
  "checkpoints": {
    "valid_count": 2,
    "steps": [50, 100],
    "latest_step": 100,
    "latest_path": "adapters/demo_run/checkpoint-100"
  },
  "final_adapter": { "present": true, "path": "adapters/demo_run" },
  "merged_model": { "present": true, "path": "merged/demo_run" },
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "resume": {
    "ready": true,
    "checkpoint_path": "adapters/demo_run/checkpoint-100"
  },
  "export": {
    "ready": true,
    "source_path": "adapters/demo_run",
    "source_kind": "final_adapter"
  }
}
```

How to read it:

- `resume.checkpoint_path` is exactly the snapshot `agoge train-qlora` would pick up with `resume_from_latest_checkpoint: true`.
- `export.source_path` is exactly what `agoge export-final-model` would merge, and `source_kind` says which kind it is: `final_adapter` means the run-root adapter, `checkpoint` means the latest valid `checkpoint-N`.
- `checkpoints.valid_count` counts only checkpoints that pass the validity rules — a `checkpoint-N` directory holding both `trainer_state.json` and a safetensors adapter. A half-written snapshot is not counted.
- `merged_model` probes the conventional `merged/<run_name>` sibling of `adapters/<run_name>`, or exactly `--merged-dir` when you pass it. `"present": false` means "not exported yet", not an error.
- Legacy `.bin` adapters read as absent and not ready under the safetensors-only policy, until `--allow-unsafe-serialization` is passed.
- The exit code is `0` for any inspectable directory, including one where nothing is ready yet. It is non-zero when the path is missing, is not a directory, contains `..`, cannot resolve a `~user` home, or inspection hits a permission/I/O failure while building the report.

The report loads no model weights, so it needs no GPU and no network.

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
