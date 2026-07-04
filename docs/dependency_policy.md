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

## Rust dependency scanning (Snyk SBOM)
Snyk Open Source does not support `snyk test` on `Cargo.toml` directly. CI generates CycloneDX SBOMs and scans them with `snyk sbom test`:

```bash
cd rust-tools
cargo install cargo-cyclonedx --locked
cargo cyclonedx --format json --override-filename sbom
find crates -name sbom.json -exec snyk sbom test --file={} --severity-threshold=medium \;
```

Generated `sbom.json` files under `rust-tools/crates/` are build artifacts — do not commit them.

## Snyk baseline policy (`.snyk`)
The Snyk GitHub App PR check (`security/snyk`) and in-repo workflows share the root `.snyk` policy file. It documents accepted-risk ignores for upstream advisories that have no fix yet (notably `transformers` and `accelerate`). Fixable transitive issues are pinned in `pyproject.toml` and `uv.lock` instead of being ignored.

`security_scan.yml` intentionally keeps pytest + Rust clippy only; Snyk jobs live in `snyk_security.yml` so security scans can be required independently in branch protection.
