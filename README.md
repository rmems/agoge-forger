# agoge-forger ⚔️

**A local-first model-training forge for modern Hugging Face models, adapter fine-tuning, quantization experiments, CUDA kernels, and optional Rust/JAX/cloud backends.**

`agoge-forger` is named after the **agoge**, the ancient Spartan training system.  
This repo is a training camp for models: disciplined, reproducible, GPU-aware, and designed to harden experimental fine-tuning workflows into reusable infrastructure.

The first target is practical:

> Fine-tune modern Hugging Face models locally with QLoRA/LoRA, save adapters, evaluate runs, and keep the path open for CUDA, Rust, JAX, and cloud-scale training.

---

## Mission

`agoge-forger` is not a chatbot app and not a one-off notebook repo.

It is a reusable forge for:

- local QLoRA / LoRA fine-tuning
- adapter training and merging
- Hugging Face dataset preparation
- model inspection and LoRA target discovery
- RTX 5080 16 GB VRAM experiments
- CUDA kernel experimentation
- optional JAX backend research
- optional Rust training/tooling integrations
- optional HCL/Terraform cloud deployment
- future Limen-Neural library integrations

Primary training path:

```text
PyTorch → Transformers → PEFT → TRL → bitsandbytes
