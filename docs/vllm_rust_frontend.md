# vLLM Rust Frontend Benchmark Lane

This repo now includes a serving/inference lane for benchmarking the vLLM Python
frontend against the newer Rust frontend. The lane is purely additive: it does not
touch training, PEFT, QLoRA, or model-loading code used for training.

## CLI commands

```bash
# Serve a model with the Python frontend
agoge serve-vllm --model HuggingFaceTB/SmolLM2-135M-Instruct --frontend python

# Serve with the experimental Rust frontend
agoge serve-vllm --model HuggingFaceTB/SmolLM2-135M-Instruct --frontend rust

# Benchmark Python vs Rust (dry-run, no GPU required)
agoge bench-vllm-frontend \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt-set configs/prompts/smoke.yaml \
  --dry-run

# Real benchmark on a GPU host (requires a working vLLM install)
agoge bench-vllm-frontend \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt-set configs/prompts/smoke.yaml \
  --out-dir runs/vllm_bench_real
```

## Config files

- `configs/serving/vllm_python_frontend.yaml`
- `configs/serving/vllm_rust_frontend.yaml`
- `configs/prompts/smoke.yaml`

You can also pass `--config <yaml>` to `serve-vllm` or `bench-vllm-frontend` and
override any field with CLI flags.

## Environment variables

- `VLLM_USE_RUST_FRONTEND=0` is set for `frontend=python`.
- `VLLM_USE_RUST_FRONTEND=1` is set for `frontend=rust`.
- `VLLM_RUST_FRONTEND_PATH` is honored by vLLM when the Rust frontend is enabled.

## Dry-run mode

`--dry-run` is the preferred way to validate the command wiring, flag parsing, and
artifact generation in CPU-only or vLLM-less environments:

```bash
agoge serve-vllm --config configs/serving/vllm_rust_frontend.yaml --dry-run
agoge bench-vllm-frontend --model <model> --prompt-set configs/prompts/smoke.yaml --dry-run
```

Dry-run does not start a vLLM server; it prints the command it would run (for
`serve-vllm`) or writes synthetic benchmark results (for `bench-vllm-frontend`).

## Real-GPU reproduction

1. Install a vLLM build that supports the Rust frontend, e.g.:

   ```bash
   pip install vllm>=0.11.0
   ```

2. Verify the Rust frontend binary is present:

   ```bash
   python - <<'PY'
   import vllm.envs
   import os
   os.environ["VLLM_USE_RUST_FRONTEND"] = "1"
   print(vllm.envs._resolve_rust_frontend_path())
   PY
   ```

3. Start a server and run the benchmark:

   ```bash
   agoge serve-vllm --config configs/serving/vllm_rust_frontend.yaml &
   agoge bench-vllm-frontend \
     --model HuggingFaceTB/SmolLM2-135M-Instruct \
     --prompt-set configs/prompts/smoke.yaml \
     --frontend rust \
     --out-dir runs/rust_frontend_real
   ```

4. Compare against the Python frontend by repeating with
   `configs/serving/vllm_python_frontend.yaml` or omitting `--frontend` to run
   both sequentially.

## Artifact layout

`bench-vllm-frontend` writes the following to the output directory:

- `results.jsonl` — one JSON object per prompt/frontend run.
- `summary.md` — human-readable report with environment, per-result table, and
  Python-vs-Rust comparison.
- `comparison.csv` — aggregated mean metrics per frontend.

The report includes the exact vLLM version, CUDA version, GPU name, and relevant
environment variables.

## Notes

- `HF_TOKEN` is never baked into configs or scripts; pass it at runtime with
  `-e HF_TOKEN=...` when running inside Docker, or export it before starting a
  server that needs gated models.
- The Rust frontend availability depends on the vLLM build. `serve-vllm` fails
  with a clear diagnostic if Rust support is requested but unavailable.
