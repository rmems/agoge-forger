# CUDA: not here

This directory is a stub. `agoge-forger` does not own a first-party CUDA
tree — see [`AGENTS.md`](../AGENTS.md).

Custom `.cu` / CUTLASS work, GPU kernel measurement, and host-scheduling
(Green Context) investigations for this host's RTX 5080 live in
[`rmems/blackwell-kernel-lab`](https://github.com/rmems/blackwell-kernel-lab).
Start there:

- [`docs/FORGE_CONSUME.md`](https://github.com/rmems/blackwell-kernel-lab/blob/main/docs/FORGE_CONSUME.md) —
  the consume-side contract: stable paths to host facts, the kernel layer
  map, runnable recipes, and build/smoke steps this project may cite.
- [`docs/FORGE_BOUNDARY.md`](https://github.com/rmems/blackwell-kernel-lab/blob/main/docs/FORGE_BOUNDARY.md) —
  the ownership rule: kernels and engine-CUDA measurement live there,
  training stays here.

Do not copy `.cu` files into this directory to "fix" this stub.
