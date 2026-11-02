# Datasets

Real training and evaluation data lives outside the repository (on the GPU host
under `~/agoge-data/`; see `AGENTS.md`). Only `samples/tiny_sft.jsonl` is tracked:
the shipped trainer configs point at it for smoke runs and `tests/test_config.py`
uses it as a fixture. Every other `samples/*.jsonl` is gitignored, and measured runs
pin their dataset and split by immutable digest rather than by path.

The accepted JSONL row formats and their validation rules are specified in
`docs/contracts/dataset_jsonl.md`.
