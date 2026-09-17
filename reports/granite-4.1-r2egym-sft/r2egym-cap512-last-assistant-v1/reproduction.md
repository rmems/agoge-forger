# Granite 4.1 G0/G1 on R2EGym SFT traces (`r2egym-cap512-last-assistant-v1`)

New experiment, not a retune of RM-760. Knobs are frozen in `experiment-contract.json`.

Last-assistant `completion_start_char` is derived from `role-capitalize-v1` dumps (line-start `Assistant: `). Exact-match scores the **final assistant turn including the role header**.

Training uses `max_seq_length: 8192`. Completion-only training **refuses** longer sequences (Granite: 121/411 train rows fit; held-out prompts 24/55 fit `context_window` 8192). Longer traces are unsupported, not silently truncated.

## Measured (2026-09-14)

| Stage | Result |
| --- | --- |
| G0 | 0/24 scored exact-match; 31/55 invalid (`g0-eval/g0-base/metrics.json`) |
| G1 preprocess | 121 rows kept, 290 dropped over 8192 (`completion_preprocessing.json`) |
| G1 train | **CUDA OOM** on first step; no adapter (`training-results.json`, `report.md`) |

## Admit and freeze

```bash
# Source: Hugging Face R2E-Gym/R2EGym-SFT-Trajectories parquet sha256
# a9b3345a097eb4961adc8a32db35db0e9105cb6272a6211c3839b19856e99cb2
# cap-512 role-capitalize admit, then last-assistant offsets:

uv run python -c "from agoge_forger.role_text import last_assistant_completion_start_char; print(last_assistant_completion_start_char.__doc__)"

uv run agoge freeze-split \
  --source ~/agoge-data/admitted/r2egym_sft_cap512_last_assistant_v1.jsonl \
  --source-path admitted/r2egym_sft_cap512_last_assistant_v1.jsonl \
  --output-dir ~/agoge-data/splits/r2egym-sft-cap512-last-assistant-v1 \
  --source-repository R2E-Gym/R2EGym-SFT-Trajectories \
  --source-revision a9b3345a097eb4961adc8a32db35db0e9105cb6272a6211c3839b19856e99cb2 \
  --dataset-version r2egym-sft-cap512-last-assistant-v1 \
  --seed 20260908 --salt r2egym-sft-cap512-last-assistant-v1
```

Pinned copy: `pinned-split/` next to this file.

## G0 (before G1)

```bash
cd /home/raulmc/rmems/agoge-forger
uv run agoge g0-held-out-eval \
  --split-manifest reports/granite-4.1-r2egym-sft/r2egym-cap512-last-assistant-v1/pinned-split/split_manifest.json \
  --output-dir reports/granite-4.1-r2egym-sft/r2egym-cap512-last-assistant-v1/g0-eval \
  --experiment-id r2egym-cap512-last-assistant-v1 \
  --base-model-id /home/raulmc/.models/ibm-granite/granite-4.1-3b-base \
  --base-revision dacb9cb9157bec98e99b09f285c92a4d58405c96 \
  --context-window 8192 --max-new-tokens 1024 --seed 17 \
  --truncation-policy mark_unsupported \
  --device-map cuda:0
```

## G1 train

```bash
uv run agoge train-qlora --config reports/granite-4.1-r2egym-sft/r2egym-cap512-last-assistant-v1/train.yaml
```

## Paired eval

```bash
uv run agoge held-out-eval \
  --split-manifest reports/granite-4.1-r2egym-sft/r2egym-cap512-last-assistant-v1/pinned-split/split_manifest.json \
  --sft-artifact /home/raulmc/agoge-data/adapters/granite-4.1-r2egym-cap512-last-assistant-v1/r2egym-cap512-last-assistant-v1 \
  --output-dir reports/granite-4.1-r2egym-sft/r2egym-cap512-last-assistant-v1/g1-eval \
  --base-model-id ibm-granite/granite-4.1-3b-base \
  --base-revision dacb9cb9157bec98e99b09f285c92a4d58405c96 \
  --context-window 8192 --max-new-tokens 1024 --seed 17 \
  --truncation-policy mark_unsupported \
  --device-map cuda:0
```
