# Hermes raw-trace custody

Operation Prometheus may ingest **deliberately supplied, bounded local Hermes
agent traces**. Personal conversations, live chat history, and opportunistic
scrapes are out of scope.

## What may be committed

- Small, invented synthetic fixtures under `tests/fixtures/hermes/`.
- Canonical trajectory v1.1 JSONL produced by
  `scripts/normalize_hermes_trajectories.py`.
- Machine-readable accepted/quarantined/rejected decision reports from that
  command.

Raw Hermes session dumps, Ollama prompt caches, and verifier working trees stay
**outside Git**. Prefer `$PROMETHEUS_DATA_ROOT/hermes/raw/` or an equivalent
local path that is already gitignored.

## Binding, not collection

The normalizer is a local-file tool. It hashes the frozen input bytes and binds
them onto `execution_provenance`:

- Hermes producer name, version, and revision
- nonempty run, session, task, and raw-trace identities plus the raw-trace
  SHA-256
- the exact admitted model identity plus the SHA-256 of the `--model-admission`
  report (`schema_version: local_model_admission_v1`, singular `decision`,
  `reasons`, model/runtime/rights/provider/fallback/probe evidence, and frozen
  `input_digests`)
- Ollama runtime identity and allowlisted, sanitized configuration
- repository base/head revisions and isolated-workspace identity from the
  frozen run manifest (conflicting trace values are rejected)
- reasoning-retention policy
- independent verifier identity, version, outcome, and artifact hashes from the
  frozen run manifest (trace self-assertions are not authoritative; conflicts
  are rejected)

It does not start Hermes, pull models, or contact a network service. Tests
inject a local-file replay boundary; HTTP URLs are refused.

## Fail closed

Duplicate JSON keys, duplicate run identities, malformed or truncated JSONL,
empty input, file-level manifest/admission errors, secret or credential
leakage (including structured keys, manifest-derived Ollama/provider fields,
and credential-bearing URL paths), home-path leakage, admission-digest
mismatch, and output/input (including hard-link) collisions never become
accepted training rows. Unverifiable runs are quarantined rather than treated
as successful. Canonical output is written as UTF-8 bytes with explicit LF.
