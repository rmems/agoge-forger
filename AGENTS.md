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
