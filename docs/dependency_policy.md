# Dependency policy

## Source of truth

`pyproject.toml` is the human-maintained declaration for Python package and
development dependencies. `uv.lock` is the committed, reproducible resolution.
Together they are the only dependency source of truth.

Do not hand-maintain `requirements.txt` or a second lock format. If an external
system requires requirements-format input, generate it mechanically from the
committed lockfile during that system's build and do not commit the generated
file.

## Updating dependencies

1. edit the smallest justified declaration in `pyproject.toml`;
2. run `uv lock`;
3. review every direct and transitive change in `uv.lock`;
4. run `uv lock --check`, the CPU quality gates, and relevant smoke tests; and
5. update the third-party license inventory for any distributed artifact.

Broad lower bounds support hardware-specific exploration. The lockfile, exact
model and dataset revisions, source commit, and run manifest provide measured
experiment reproducibility. Do not present an unpinned exploratory environment
as a reproduced result.

## Docker image

The CPU/smoke Dockerfile installs the project from the committed lockfile:

```dockerfile
RUN uv venv /app/.venv \
    && uv sync --no-dev --no-editable --locked
```

- `--no-dev` excludes test and lint tools from the runtime image.
- `--no-editable` installs the project as a wheel.
- `--locked` fails when `pyproject.toml` and `uv.lock` disagree.

The image must not bake in tokens or credentials. Pass secrets only at runtime.
See `THIRD_PARTY_NOTICES.md` before distributing the image.

## Security review

Dependabot and the repository's security tooling help identify dependency
updates and known advisories; they do not determine whether a model, dataset, or
package may be redistributed. Fixable transitive issues belong in the declared
dependency graph and lockfile. Temporary accepted risks must be documented with
scope, rationale, and an expiry/review condition.
