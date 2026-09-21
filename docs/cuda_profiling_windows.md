# Bounded CUDA profiling windows (opt-in)

This is **not** part of ordinary Agoge training. The default MiniCPM5 canary
and `agoge train-qlora` without `--profile-window` / `AGOGE_PROFILE_WINDOW`
never start a profiler.

Custom CUDA kernels, Nsight recipes, and CUPTI collectors stay in
[`rmems/blackwell-kernel-lab`](https://github.com/rmems/blackwell-kernel-lab).
Agoge only:

1. Emits phase/step markers (`docs/contracts/f0_correlation_markers.md`).
2. Accepts a window request as `run_id` + phase + inclusive step range.
3. Optionally runs **torch.profiler** for that window and writes a compact
   summary that can be joined to BKL GPU samples.

## Recipe (ShipOfTheseus / MiniCPM5)

The canary YAML uses `gradient_accumulation_steps: 8` and ten tiny rows, so
it typically completes **one optimizer step**. Profile that step as the
steady-state forward/backward/optimizer sequence:

```bash
export AGOGE_RUN_ID=run_minicpm5_profile_$(date -u +%Y%m%dT%H%M%SZ)
export AGOGE_PROFILE_WINDOW=train:1-1

# Optional: BKL GPU sampler from blackwell-kernel-lab, same AGOGE_RUN_ID.
# Do not start Nsight from this repo.

uv run agoge train-qlora --config configs/minicpm5_canary.yaml
```

Outputs (gitignored under `runs/`):

- `runs/<run_name>/telemetry/agoge-markers.jsonl`
- `runs/<run_name>/telemetry/profile-window-request.json`
- `runs/<run_name>/telemetry/profile-window.json`

For a multi-step window (warmup step plus a profiled step), set
`gradient_accumulation_steps: 1` in a local copy of the canary and use
`train:2-2` so step 1 remains an unprofiled baseline for overhead.

Join to BKL GPU JSONL with the same `agoge_run_id` and
`profile_window_id` / `profile_window_ref`. Timing uses `monotonic_ns` on one
host; assume wall clocks stay within 1 s.

## Overhead (measured)

`torch.profiler` is the in-process backend. Each profiled run writes `overhead`
into `profile-window.json`:

- `profiled_step_mean_s` / `unprofiled_step_mean_s` / `ratio` when the run has
  both profiled and unprofiled optimizer steps (`status: ok`).
- `status: unavailable` when every step was inside the window (typical MiniCPM5
  canary: one optimizer step) or the profiler never started. Missing is not
  encoded as ratio `0`.

A separate CPU matmul microbenchmark
(`agoge_forger.telemetry.overhead.measure_torch_profiler_overhead`) exists so
CI can prove the profiler path runs without a GPU. Its wall-time ratio on a
tiny loop is **not** a training-quality claim; use the per-run step comparison
on ShipOfTheseus.

Nsight/CUPTI overhead is a BKL measurement. This repo does not launch `nsys` /
`ncu`. Ordinary training without a window does not start torch.profiler;
marker JSONL appends are microseconds per logged step.

## Profiler permissions and tools

Keep these off the default training path:

| Backend | Agoge behavior | Prerequisites if BKL attaches it |
| --- | --- | --- |
| `torch` | Used when a window is requested with `backend: torch` (default). | PyTorch with CUDA for kernel/sync/transfer evidence. CPU-only hosts record CUDA fields as `unavailable`. |
| `nsight_systems` | `unsupported` (not invoked). | `nsys` on PATH; often `perf_event_paranoid` ≤ 2 or CAP_SYS_ADMIN / CAP_PERFMON for some events. |
| `nsight_compute` | `unsupported` (not invoked). | `ncu`; kernel replay can distort timings and VRAM. |
| `cupti` | `unsupported` (not invoked). | CUPTI libraries matching the driver; typically a BKL wrapper, not a trainer hook. |

Do not chmod `/proc/sys/kernel/perf_event_paranoid` for a normal SFT run.

## What the compact summary keeps

When torch.profiler can see the device:

- kernel names / families (attention, GEMM, transfer, sync, allocator, other)
- duration p50/p95/sum (per-event when kineto exposes `events()`, otherwise
  key-average means with `distribution: key_averages_mean_only`)
- sync and H2D/D2H transfer evidence when those names appear
- CUDA Graph capture/replay: `unavailable` (not a first-class torch.profiler flag)
- allocator/memory events when `profile_memory=True` surfaces them
- profiler/backend version

Missing evidence uses `status: unavailable` or `unsupported` with `value: null`.
Zero is only stored for a real measured zero.

Raw chrome traces, if you export them locally, stay under `runs/` and must
not be committed.
