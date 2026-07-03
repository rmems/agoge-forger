# Python, Rust, and Julia Smoke Test Workflows

This project includes three manual GitHub Actions workflows for smoke testing across Python, Rust, and Julia toolchains.

## Available Workflows

| Workflow | File | Language |
|----------|------|----------|
| Python Smoke Test | `.github/workflows/python_smoke_test.yml` | Python |
| Rust Smoke Test | `.github/workflows/rust_smoke_test.yml` | Rust |
| Julia Smoke Test | `.github/workflows/julia_smoke_test.yml` | Julia |

## Launching a Workflow

1. Go to the **Actions** tab in the GitHub repository.
2. Select the workflow from the left sidebar.
3. Click **Run workflow** (top right).
4. Fill in the input parameters.
5. Click **Run workflow** to start.

All three workflows are **manual-only** (`workflow_dispatch`). They will never trigger automatically on push or PR events.

## Python: Python Smoke Test

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `model` | `HuggingFaceM4/tiny-random-LlamaForCausalLM` | Hugging Face model ID |
| `workload` | `inference` | Workload type: `inference`, `eval`, `inspect` |
| `max_requests` | `5` | Maximum inference requests |
| `concurrency` | `1` | Request concurrency |
| `stream` | `false` | Enable streaming |
| `dry_run` | `false` | Dry-run mode (no real inference) |
| `upload_artifacts` | `true` | Upload run artifacts |

### Artifacts

- `manifest.json` — Experiment config snapshot
- `provider.json` — Runtime environment info
- `usage_before.json` — Token-usage snapshot before the run
- `usage_after.json` — Token-usage snapshot after the run
- `usage_delta.json` — Usage difference (after − before)
- `results.jsonl` — Per-request results (one JSON object per line)
- `summary.md` — Human-readable summary

### Dry-Run Mode

Set `dry_run` to `true` to validate the workflow without making real inference calls. All requests will report `status: dry_run` with zero tokens and latency.

## Rust Smoke Test

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `command` | `check` | Cargo command: `check`, `test`, `clippy`, `fmt`, `build` |
| `package` | (empty) | Specific crate (leave empty for full workspace) |
| `dry_run` | `false` | Validate Cargo.toml only, no compilation |
| `upload_artifacts` | `true` | Upload run artifacts |

### Dry-Run Mode

When `dry_run` is `true`, the workflow validates the workspace structure and `Cargo.toml` without compiling. It lists workspace members and confirms the project is valid.

## Julia Smoke Test

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `command` | `check` | Julia command: `check`, `test`, `build` |
| `project_dir` | `.` | Julia project directory |
| `julia_version` | `1.10` | Julia version to install |
| `dry_run` | `false` | Validate Julia version only |
| `upload_artifacts` | `true` | Upload run artifacts |

### Dry-Run Mode

When `dry_run` is `true`, the workflow only verifies that the requested Julia version is installed and functional.

## Local Testing

Run the Python smoke test locally:

```bash
python scripts/smoke_test.py \
  --model HuggingFaceM4/tiny-random-LlamaForCausalLM \
  --workload inference \
  --max-requests 3 \
  --concurrency 1 \
  --dry-run \
  --output-dir smoke_output
```

Run the Rust smoke test locally:

```bash
cd rust-tools && cargo check
cd rust-tools && cargo test
cd rust-tools && cargo clippy -- -D warnings
```

Run the Julia smoke test locally:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

## Security Scanning (Snyk + Aikido)

Two workflows run automatically on pull requests when matching paths change:

| Workflow | File | Triggers on |
|----------|------|-------------|
| Snyk Security | `.github/workflows/snyk_security.yml` | `src/`, `scripts/`, Python manifests, `rust-tools/`, `infra/` |
| Aikido Security | `.github/workflows/aikido_security.yml` | `src/`, `scripts/`, `rust-tools/`, `infra/`, security workflows |

Both also support manual runs via **workflow_dispatch**.

### Repository secrets

| Secret | Workflow | Source |
|--------|----------|--------|
| `SNYK_TOKEN` | Snyk Security | [Snyk account settings](https://app.snyk.io/account) |
| `AIKIDO_API_KEY` | Aikido Security | [Aikido Local Scanner setup](https://app.aikido.dev/settings/integrations/localscan) |

When a secret is unset, the corresponding scan steps are skipped so forks and unconfigured repos still pass.

### Snyk jobs

| Job | Scan | Blocking? |
|-----|------|-----------|
| `snyk-python-sca` | `uv.lock` (fallback: `requirements.txt`) | No (`continue-on-error`) — transformers advisories may have no upstream fix |
| `snyk-python-code` | SAST on `src/`, `scripts/` + SARIF upload | Yes (high+) |
| `snyk-rust` | SAST on `rust-tools/` + CycloneDX SBOM → `snyk sbom test` + SARIF | Yes |
| `snyk-iac` | Terraform under `infra/terraform/` | Yes (medium+) |

**Plan notes:** `uv.lock` SCA and Rust Snyk Code may require Snyk Enterprise Early Access. Do not use `snyk test --all-projects` — it picks up unrelated manifests (e.g. `.kilo/`).

### Aikido jobs

| Job | When | Behavior |
|-----|------|----------|
| `aikido-pr-gate` | `pull_request` | Local Scanner PR gating (`--gating-mode pr`, `--fail-on critical`) — only **new** issues fail |
| `aikido-baseline` | `workflow_dispatch` | Full-repo scan on the selected branch (seed comparison baseline) |

**Alternative (no workflow YAML):** Install the [Aikido PR Checks GitHub App](https://help.aikido.dev/pr-and-release-gating/github-ci-pr-gating-via-aikido-dashboard) and configure gating from the Aikido dashboard.

### Local commands

```bash
# Snyk (after uv pip install -e ".[dev]")
snyk test --file=uv.lock --package-manager=uv
snyk code test src/ scripts/ --severity-threshold=high
snyk code test rust-tools/ --severity-threshold=high
snyk iac test infra/terraform --severity-threshold=medium
cd rust-tools && cargo install cargo-cyclonedx --locked
cargo cyclonedx --format json --override-filename sbom
find crates -name sbom.json -exec snyk sbom test --file={} \;

# Aikido — use IDE MCP during development (.cursor/rules/aikido_rules.mdc)
# CI uses the Local Scanner container; see aikido_security.yml for flags.
```

### Branch protection (optional)

After baselines are clean, require these status checks under branch protection:

- `Snyk Security / snyk-python-code`
- `Snyk Security / snyk-rust`
- `Snyk Security / snyk-iac`
- `Aikido Security / aikido-pr-gate`

---

*Agent: Kilo agent OpenCode Go/MiniMax-M3*
