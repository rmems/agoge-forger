# Granite 4.1 G0 → G1 on R2EGym (`r2egym-cap512-last-assistant-v1`)

**Conclusion:** **G1 did not complete.** G0 ran under the frozen eval contract. QLoRA did not produce an adapter, so there is no paired G1 exact-match result. This is not a retune of RM-760.

Frozen contract: `experiment-contract.json`. Knobs were not changed after G0.

## G0 (base-only, before training)

- Bundle: `g0-eval/g0-base/`
- Held-out tasks: 55; **24 scored**, **31 invalid** (`mark_unsupported` when the prompt does not fit `context_window` 8192)
- Base exact-match accuracy: **0.0000** (0/24)

## G1 (one bounded QLoRA epoch)

- Preprocessing completed: **121/411** train rows fit `max_seq_length` 8192 (290 dropped as over-budget; no truncation). Evidence: `completion_preprocessing.json` (also under the adapter `output_dir`).
- Training **failed at step 0/16** with `torch.OutOfMemoryError` on RTX 5080 (16 GB). 4-bit Granite 4.1 3B + LoRA + `max_seq_length` 8192 does not fit this host for a backward pass. Allocator retry (`expandable_segments:True`) also OOM.
- No `adapter_model.safetensors` was written. Do not lower `max_seq_length` or eval windows on this experiment id after seeing that result.

## Paired held-out eval

Not run. `agoge held-out-eval` requires an SFT artifact.

## Notes

- Last-assistant `completion_start_char` is the line-start of the final `Assistant: ` header. Exact-match includes that header.
- Hugging Face `datasets` cache can serve stale JSONL; use a fresh `HF_DATASETS_CACHE` if completion offsets were rewritten on disk.
- Unload Ollama embedding models before GPU jobs on this machine.
