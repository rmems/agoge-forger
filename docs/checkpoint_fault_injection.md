# Checkpoint fault injection

This repository proves single-process checkpoint recovery on CPU. The
bounded fake trainer in `agoge_forger.train.fault_harness` does not download
models, allocate CUDA memory, or start a distributed job. GitHub CI runs that
path only.

The harness writes `checkpoint-N` trees through the same atomic no-replace
directory publish used by eval bundles and split snapshots. Incomplete trees
are moved under `.agoge-quarantine/` with `quarantine_reason.json`. Resume
selects the last complete checkpoint and reports `equivalent=true` only when
optimizer, scheduler, global step, sampler position, and RNG state are all
present.

```bash
uv run pytest tests/test_checkpoint_fault_harness.py -q
```

Injected faults cover OOM, SIGINT/SIGTERM-equivalent interruption, a short
weight write, rename failure, and final-adapter export failure. Failure
cleanup keeps the last complete checkpoint and the quarantine record; it does
not call `cleanup-run` and does not hide repeated OOMs by changing
hyperparameters. This work is independent of multi-GPU training.

## Optional trusted-GPU recipe

Do not run this in CI. On a trusted local GPU host (the RTX 5080 notes in
[`rtx5080_local_training.md`](rtx5080_local_training.md)), the CPU harness
already covers the recovery contract. Optional live evidence is a real
single-GPU run that you interrupt yourself:

1. Start a canary QLoRA job that writes checkpoints under `adapters/<run_name>`.
2. Interrupt it with `SIGINT` or `SIGTERM` during a step, or let the process
   abort on CUDA OOM. Do not retune batch size or sequence length to hide the
   fault.
3. Inspect the run without loading weights:

   ```bash
   uv run agoge run-status adapters/<run_name> --format table
   ```

4. Confirm `resume_ready` points at the highest complete `checkpoint-N`, not a
   partial tree. Incomplete `checkpoint-*` directories should be absent from
   that latest path; after `agoge train-qlora` resume they are quarantined
   with an explicit reason under `adapters/<run_name>/.agoge-quarantine/`.
5. Resume the same config. `run-status` must keep reporting the restored
   checkpoint until a newer complete snapshot exists. If optimizer,
   scheduler, or RNG files are missing, `resume_ready` stays `no` and the
   process must not be treated as equivalent to a clean stop.

Keep the interrupted tree, `run-status` JSON, and quarantine reason files as
the GPU evidence pack. Do not delete checkpoints with `cleanup-run` until a
final adapter or merged model is present.
