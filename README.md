# Agoge Forger

"Agoge" refers to the ancient Spartan training system. This repo is a modern model-training forge: local-first, GPU-aware, cloud-capable, and designed for fine-tuning, adapter training, quantization experiments, CUDA kernel experimentation, and optional Rust/JAX backends.

## Philosophy

- **PyTorch Primary**: PyTorch is the main training engine. It has the ecosystem, the tools (PEFT, TRL, BitsAndBytes), and the community.
- **JAX Optional**: Included as a stubbed backend for future algorithmic research and specialized workloads.
- **Rust Optional**: Experimental tooling, CLI enhancements, and future framework integrations.
- **RTX 5080 16GB Focus**: Local defaults are tuned to fit within 16GB of VRAM using QLoRA (NF4, double quant) and sequence lengths of 2048.
- **Full Fine-Tuning**: Possible, but not the local default. QLoRA/LoRA are preferred.

## Quickstart (Smoke Test)

### 1. Check Environment
```bash
make setup
agoge check-torch
```

### 2. Inspect an Architecture (e.g. GKA-HQwen3)
Before training custom architectures, inspect their structure to find LoRA targets:
```bash
agoge inspect-model --model-id amazon/GKA-primed-HQwen3-8B-Reasoner
agoge inspect-lora-targets --model-id amazon/GKA-primed-HQwen3-8B-Reasoner
```

### 3. Run QLoRA Smoke Test
Runs a tiny JSONL dataset against a base model to verify PEFT saving and memory:
```bash
make train-smoke
# Or manually: agoge train-qlora --config configs/smoke_test.yaml
```

### 4. Evaluate Adapter
```bash
make eval-smoke
# Or manually: agoge smoke-eval --adapter-path adapters/<run_name>
```

### 5. Merge Adapter
```bash
agoge merge-adapter --base-model <base_model> --adapter-path adapters/<run_name> --out-dir merged/<run_name>
```

### 6. Resume From The Latest Valid Checkpoint

```bash
agoge train-qlora --config configs/my_run.yaml
```

Set these config fields to keep checkpoint growth under control and resume automatically:

```yaml
save_steps: 50
save_total_limit: 2
resume_from_latest_checkpoint: true
```

**Release note:** training now defaults to `save_strategy="steps"` (previously `"no"`) so `--resume` and `resume_from_latest_checkpoint` work out of the box. Use `save_total_limit` to cap retained `checkpoint-*` trees and limit disk growth.

### 7. Export One Final Model Artifact

Use this when a run has multiple `checkpoint-*` snapshots and you want one merged deliverable:

```bash
agoge export-final-model --run-dir adapters/<run_name> --out-dir merged/<run_name>
```

`checkpoint-*` directories are trainer recovery snapshots. The adapter saved at `adapters/<run_name>` is the LoRA output for continued PEFT work. The merged model under `merged/<run_name>` is the single final artifact to ship or evaluate as a standalone model.

### 8. Serve and Smoke with vLLM

Serve the merged model with vLLM, then run a compatibility smoke test:

```bash
agoge serve-vllm --model merged/<run_name>
agoge smoke-vllm --model merged/<run_name> --run-name smoke_<run_name>
```

See `docs/vllm_model_compatibility.md` for the full compatibility matrix and
`docs/chat_completions_providers.md` for `agoge smoke-vllm` options.

## Docker

A CPU/smoke Docker image is available at the repository root:

```bash
docker build -t agoge-forger:local .
docker run --rm agoge-forger:local agoge --help
docker run --rm agoge-forger:local python -c "import agoge_forger; print(agoge_forger.__version__)"
```

The image installs the package from `uv.lock` with `--no-dev --no-editable`, runs as a non-root `app` user, and does **not** bake in `HF_TOKEN` or other secrets. Pass `HF_TOKEN` at runtime if a command needs the Hugging Face Hub.

## Cloud Infrastructure
The `infra/` folder contains Terraform scaffolds for AWS, Azure, DigitalOcean, and IBM Cloud. They are stubs for future cloud-scale training using HCL.

## GGUF Notes
GGUF conversion is *not* automatic, especially for custom architectures like GKA or SSMs. Llama.cpp must explicitly support the architecture before GGUF export will work. See `src/agoge_forger/export/gguf_notes.py`.

## Second-pass safety guarantees

This forge is designed to protect your environment and artifacts:
- **Remote code opt-in**: `trust_remote_code` defaults to `false`. Set `trust_remote_code: true` in YAML or pass `--trust-remote-code` only for model repos you explicitly trust.
- **No Hallucinated Configs**: Model configurations cannot hardcode unknown LoRA targets. `target_modules` must be validated against the inspected model graph before training.
- **Safetensors Default**: Safetensors is the default and required save format. The forge will fail if `.bin` files are generated, preventing execution vulnerabilities.
- **Path validation**: Config, dataset, adapter, checkpoint, and output paths reject `..` traversal before use.
- **Artifact Indexing**: All output directories generate an `artifact_index.json` containing the sizes and SHA256 hashes of the adapters/shards.
- **Reproducible Manifests**: Every training run generates a comprehensive `manifest.json` including Git state, environment versions, tokenizer metadata, and exact GPU telemetry.
- **Safe Metadata Inspection**: Model architecture metadata can be inspected entirely without downloading large weights using `agoge model-metadata`.
- **Checkpoint Hygiene**: Training saves checkpoints on a fixed step interval, caps retained checkpoint trees, warns when local disk headroom is thin, and can resume from the latest valid checkpoint automatically.

## Security scanning

- **Dependabot** watches Python, GitHub Actions, and Rust dependencies (`.github/dependabot.yml`).
- **PR quality gate** (`.github/workflows/pr_quality_gate.yml`): `ruff check`, `ruff format --check`, `mypy src/agoge_forger`, and `pytest tests/` on pull requests that touch Python sources (no secrets required).
- **Security scan** runs path-gated pytest and Rust checks on pull requests (`.github/workflows/security_scan.yml`).
- **Aikido PR gating** via the [Aikido PR Checks GitHub App](https://help.aikido.dev/pr-and-release-gating/github-ci-pr-gating-via-aikido-dashboard) (status: `Aikido Security: check code`).
- **Aikido release gating** (`.github/workflows/aikido_security.yml`, manual): `@aikidosec/ci-api-client scan-release`. Requires `AIKIDO_CLIENT_API_KEY` (or `AIKIDO_API_KEY`).
- **Aikido MCP** (local dev): Enable the IDE issue feed at [Aikido MCP permissions](https://app.us.aikido.dev/settings/integrations/ide/mcp/permissions), then use `/aikido:setup` in Cursor.
- **Snyk** is not run in CI (plan usage exhausted). Optional local CLI only if you still have a Snyk account.
