# RTX 5080 16GB Local Training

The RTX 5080 has 16GB of VRAM. This is a very capable card, but it has strict limits for modern 7B-8B parameter LLMs.

For anything below the training loop itself — kernel-level or engine-CUDA
behavior on this card (FlashAttention, CUDA graphs, quantization, Green
Context host scheduling) — see
[`rmems/blackwell-kernel-lab`](https://github.com/rmems/blackwell-kernel-lab),
this host's kernel source of truth, starting with
[`docs/FORGE_CONSUME.md`](https://github.com/rmems/blackwell-kernel-lab/blob/main/docs/FORGE_CONSUME.md).
This project does not duplicate that VRAM/kernel math; the preflight checks
below cover training-time fit only.

## Measured on this host

Peak VRAM and headroom for the MiniCPM5 QLoRA canary on ShipOfTheseus
(RTX 5080, `sm_120`, ~16 GB; keep ≥2 GiB free) live in the kernel lab:

[`blackwell-kernel-lab/docs/TRAIN_FIT_5080.md`](https://github.com/rmems/blackwell-kernel-lab/blob/main/docs/TRAIN_FIT_5080.md)

That page is the concise train-fit summary (dated MiniCPM5 peak-VRAM row;
Granite 4.1 seq-2048 remains an unmeasured template). Trainer YAML in this
repo stays the normative knobs (`configs/minicpm5_canary.yaml`,
`configs/granite_4_1_flagship.yaml`). Do not treat BKL L1 prefix/KV or L2
Green Context numbers as train hyperparameters.

Until `TRAIN_FIT_5080.md` is on BKL `main`, the draft is
[blackwell-kernel-lab#57](https://github.com/rmems/blackwell-kernel-lab/pull/57).

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
