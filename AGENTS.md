# Contributor guidance

## Scope

Agoge Forger is a Python/PyTorch post-training, evaluation, and release
repository. Preserve the ownership boundaries in
`docs/repository_boundaries.md`: do not add custom CUDA, alternate training
runtimes, synthetic-data implementation, or cloud-infrastructure trees here.

## Changes

- Keep trainer and evaluator changes separable from metadata, cleanup, dataset,
  and release changes.
- Treat `pyproject.toml` as the dependency declaration and `uv.lock` as its
  committed resolution. Regenerate and review the lockfile whenever dependency
  declarations change.
- Keep remote model code disabled by default. A change that enables it requires
  a documented architecture-specific gate and review.
- Pin model and dataset revisions for measured experiments. Smoke fixtures are
  plumbing checks and cannot support quality claims.
- Preserve manifest, schema, and artifact compatibility unless the issue
  explicitly authorizes a versioned contract change.
- Do not commit credentials, generated training outputs, caches, IDE metadata,
  compiled objects, or build directories.

## Validation

Run the checks relevant to the change, expanding to the complete CPU gate before
integration:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/agoge_forger
uv run pytest tests/
uv build
```

Training changes should include dataset/config validation and a tiny,
non-claiming SFT smoke when compatible hardware is available. Otherwise,
record the hardware limitation explicitly; CPU CI is not evidence that CUDA,
bitsandbytes, or vLLM paths work.

## Licensing

Contributions are submitted under Apache-2.0 unless explicitly stated
otherwise. Before adding or redistributing third-party material, follow
[the third-party notice policy](THIRD_PARTY_NOTICES.md) and record authoritative
license provenance.
