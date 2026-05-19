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

## Cloud Infrastructure
The `infra/` folder contains Terraform scaffolds for AWS, Azure, DigitalOcean, and IBM Cloud. They are stubs for future cloud-scale training using HCL.

## GGUF Notes
GGUF conversion is *not* automatic, especially for custom architectures like GKA or SSMs. Llama.cpp must explicitly support the architecture before GGUF export will work. See `src/agoge_forger/export/gguf_notes.py`.
