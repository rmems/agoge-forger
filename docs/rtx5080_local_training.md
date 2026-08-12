# RTX 5080 16GB Local Training

The RTX 5080 has 16GB of VRAM. This is a very capable card, but it has strict limits for modern 7B-8B parameter LLMs.

## Preflight Checks
Agoge-Forger runs preflight checks to warn about likely Out-Of-Memory (OOM) scenarios on cards reporting **≤16.5 GiB** total VRAM (binary GiB, same unit as disk preflight — a true 16 GiB RTX 5080 reports ~16.0, not ~17.18 decimal GB).

When that gate applies:
- It will warn if `load_in_4bit` is off (full-precision / non–4-bit loads are likely to OOM).
- It will warn if your `batch_size > 1` (use `gradient_accumulation_steps` instead).
- It will warn if `max_seq_length > 2048` (no separate model-size filter; the check is VRAM-only).

## The QLoRA Path
The local default is QLoRA:
- **NF4 Quantization**: Loads the base model in 4-bit NormalFloat.
- **Double Quantization**: Compresses the quantization constants to save more memory.
- **Paged Optimizers**: Allows offloading optimizer states to CPU RAM if needed (configured via bitsandbytes).
- **Gradient Checkpointing**: Trades compute for memory by dropping intermediate activations and recomputing them during the backward pass.
