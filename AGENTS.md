# Agoge Forger contributor guidance

Agoge Forger is a Python/PyTorch training, evaluation, and release project.

- Keep packaging metadata in `pyproject.toml` and the resolved dependency graph in
  `uv.lock`; do not add a second requirements file.
- Keep `trust_remote_code: false` in shipped model configs unless a reviewed,
  documented exception is necessary.
- Do not introduce first-party CUDA, Rust, Julia, JAX, Terraform, or cloud
  implementation trees. Integrations belong in their dedicated sibling projects.
  Custom CUDA / GPU kernel investigations for the local RTX 5080 belong in
  [`rmems/blackwell-kernel-lab`](https://github.com/rmems/blackwell-kernel-lab)
  (see `cuda/README.md` and `docs/rtx5080_local_training.md`), not here.
- Run `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy src/agoge_forger`, and `uv run pytest tests/ -q` before proposing a
  change. Use the Docker smoke-test skill when a container change needs validation.

## Learned User Preferences

- Keep pull requests small and scoped; do not expand cleanup or feature work, or open extra PRs, without explicit approval.
- Do not close GitHub issues or pull requests unless explicitly asked.
- After committing on a working branch, push to the remote; do not leave local-only commits.
- Do not widen analyzer or security excludes without approval; do not skip Qlty. Codacy and Qodana stay.
- On this GPU host, unload competing processes such as Ollama before training or eval, and use the largest context the 16GB RTX 5080 can actually run.
- Prefer Hugging Face transformers for local generate/eval; do not add a first-party vLLM stack here.
- When posting GitHub issue, PR, or review comments, cite yourself as **Cursor agent**.

## Learned Workspace Facts

- Local training worker is an RTX 5080 (16GB); data, frozen splits, adapters, and eval bundles live under `~/agoge-data/`.
- Granite 4.1 3B base is cached at `~/.models/ibm-granite/granite-4.1-3b-base`.
- Code-repair training exports come from sibling repo `rmems/synthetic-factory`.
- The flagship measured comparison is G0 (untouched Granite 4.1 3B base) versus G1 (same pinned base plus one frozen code-repair SFT); other arms stay parked until that result is publishable.
- Frozen split membership must not be re-split; measured runs pin dataset and split by immutable digest.
- Shipped configs such as `configs/granite_4_1_flagship.yaml` must not contain host-specific paths.
- Measured SFT uses completion-only loss and must evaluate G0 before G1 training.
- R2E-Gym SFT datasets are already downloaded locally for Hugging Face training.
