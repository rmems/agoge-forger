# RTX 5080 16GB Local Training

The RTX 5080 has 16GB of VRAM. This is a very capable card, but it has strict limits for modern 7B-8B parameter LLMs.

## Preflight Checks
Agoge-Forger runs preflight checks to warn about likely Out-Of-Memory (OOM) scenarios.
- It will warn if you try to train an 8B model without 4-bit quantization.
- It will warn if your `batch_size > 1` (use `gradient_accumulation_steps` instead).
- It will warn if `max_seq_length > 2048` on models larger than 3B parameters.

## The QLoRA Path
The local default is QLoRA:
- **NF4 Quantization**: Loads the base model in 4-bit NormalFloat.
- **Double Quantization**: Compresses the quantization constants to save more memory.
- **Paged Optimizers**: Allows offloading optimizer states to CPU RAM if needed (configured via bitsandbytes).
- **Gradient Checkpointing**: Trades compute for memory by dropping intermediate activations and recomputing them during the backward pass.
