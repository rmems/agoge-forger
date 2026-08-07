# Dependency Policy

## Broad Ranges for Exploration
The `pyproject.toml` uses broad `>="` version ranges to maximize compatibility across different hardware architectures and base PyTorch versions.

## Locking for Reproducibility
When moving from exploration to reproducible training, you should lock your dependencies. 
If you are using `uv`, you can generate a lockfile:
```bash
uv pip compile pyproject.toml -o requirements-lock.txt
```

## Rust Reproducibility
The Rust workspace (`rust-tools/`) commits `Cargo.lock` to ensure CLI tooling builds reproducibly across environments. Do not add `Cargo.lock` to `.gitignore`.

## Rust dependency scanning

Snyk CI was removed after the private-test plan limit was exhausted. Prefer Dependabot for Rust (`rust-tools/Cargo.lock`) and Aikido PR gating for merge checks.

If you re-enable Snyk later, Open Source does not support `snyk test` on `Cargo.toml` directly — generate CycloneDX SBOMs and scan with `snyk sbom test`:

```bash
cd rust-tools
cargo install cargo-cyclonedx --locked
cargo cyclonedx --format json --override-filename sbom
find crates -name sbom.json -exec snyk sbom test --file={} --severity-threshold=medium \;
```

Generated `sbom.json` files under `rust-tools/crates/` are build artifacts — do not commit them.

## Docker image dependency policy

The CPU/smoke `Dockerfile` builds the runtime image using the locked `uv.lock` file:

```dockerfile
RUN uv venv /app/.venv \
    && uv sync --no-dev --no-editable --locked
```

- `--no-dev` keeps test/lint tools out of the runtime image.
- `--no-editable` installs `agoge_forger` as a wheel so the image does not depend on the source tree at runtime.
- `--locked` requires `uv.lock` to be up-to-date with `pyproject.toml`; the build fails if they are inconsistent, so a stale lockfile cannot silently install outdated or incomplete dependencies.

This is a stricter, image-specific install than the host CI dev bootstrap, which runs `uv pip install -e ".[dev]"` and does not consume `uv.lock`. The Docker build enforces the lockfile so the runtime image is reproducible from a known dependency graph. Container images are not rebuilt to chase latest upstream releases; fixes are brought in by updating `pyproject.toml` and regenerating `uv.lock`.

## Snyk baseline policy (`.snyk`)
The root `.snyk` file remains for optional local CLI use. It documents accepted-risk ignores for upstream advisories that have no fix yet (notably `transformers` and `accelerate`). Fixable transitive issues are pinned in `pyproject.toml` and `uv.lock` instead of being ignored.

`.github/workflows/security_scan.yml` runs path-gated `pytest`, `cargo check`, and `cargo clippy`. In-repo Snyk workflows are disabled.
