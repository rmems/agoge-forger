---
name: Testing the agoge-forger Docker smoke image
description: How to build and validate the root CPU/smoke Docker image for agoge-forger end-to-end.
---

# Testing the agoge-forger Docker smoke image

The repo root defines a CPU/smoke `Dockerfile` and `.dockerignore`. The canonical end-to-end validation sequence is:

```bash
cd /path/to/agoge-forger

docker build -t agoge-forger:local .

docker run --rm agoge-forger:local agoge --help

docker run --rm agoge-forger:local python -c "import agoge_forger; print(agoge_forger.__version__)"

docker run --rm agoge-forger:local agoge check-torch

docker history --no-trunc agoge-forger:local | grep -i HF_TOKEN

docker run --rm agoge-forger:local /bin/sh -c 'env | grep -i hf || echo no HF env'
```

## What success looks like

- `docker build` exits `0` and tags `agoge-forger:local`.
- `agoge --help` shows the Typer usage and full command table including `check-torch`.
- `python -c "import agoge_forger; print(agoge_forger.__version__)"` prints `0.1.0`.
- `agoge check-torch` on a CPU-only host prints `PyTorch Version: ...` and `WARNING ... CUDA is NOT available. PyTorch will use CPU.`, then exits `0`. No `Device Name:` or `Total VRAM:` lines should appear.
- `docker history --no-trunc agoge-forger:local | grep -i HF_TOKEN` returns no lines.
- The `env | grep -i hf` command prints `no HF env`.

## Adversarial tips

- Docker cache is reused whenever the build context (`Dockerfile`, `.dockerignore`, and every file copied into the image) has not changed. That is a normal `docker build` outcome, but to force a true rebuild use `--no-cache`.
- `HF_TOKEN` must never be passed as a build arg or `ENV` in the `Dockerfile`. It should only be supplied at runtime with `-e HF_TOKEN=...` when a command genuinely needs the Hugging Face Hub.
- If `agoge check-torch` fails to import `torch`, first check that the build completed and `uv sync` installed the locked wheels. If `uv sync` succeeded, the failure is likely a runtime issue such as an incompatible wheel, a missing runtime library, or an architecture mismatch, not a build-time resolution problem.
- `agoge-forger` is installed as a non-editable wheel inside `/app/.venv` by the `Dockerfile`; do not expect `src` changes on the host to be reflected without a rebuild.
