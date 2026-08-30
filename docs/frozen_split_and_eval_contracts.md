# Frozen split and evaluation contracts

Agoge freezes source-level task identity before any model-specific rendering or
tokenization. `SplitManifest` in `agoge_forger.split_contract` is the one
authoritative runtime schema. Its version is `agoge.split-manifest.v1`.

## Curated source requirements

The input is a local JSONL file from an explicitly named repository, immutable
revision, and dataset version. Every source record must contain:

- `canonical_id`: globally stable sample identity;
- `lineage_id`: identity shared by variants that must remain together; and
- training content accepted by Agoge, such as `text`, `messages`, or
  `instruction`/`output`.

`group_id` is optional. When present, the complete group remains atomic. Exact
canonical content, lineage, and declared-group relationships are transitively
joined before split assignment. Identity field names can be overridden for a
versioned source that already has equivalent fields.

## Materialize one frozen snapshot

From an installed checkout:

```bash
uv run python scripts/freeze_split.py \
  --source /path/to/curated.jsonl \
  --output-dir /path/to/new-snapshot \
  --source-repository rmems/synthetic-factory \
  --source-revision <immutable-commit> \
  --dataset-version <version> \
  --seed 20260830 \
  --salt <versioned-policy-salt>
```

The output directory must not exist. Agoge refuses to overwrite or silently
regenerate a frozen snapshot. The command writes:

```text
new-snapshot/
  splits/
    train.jsonl
    validation.jsonl
    held_out.jsonl
  split_manifest.json
  split_report.md
```

The report states source coverage, record counts, partition digests,
deterministic leakage guarantees, exclusions, and limitations. Semantic
near-duplicate detection is deliberately not claimed by the deterministic
gate.

## Validate and consume without re-splitting

`validate_split_manifest(path, source_path=...)` verifies the source SHA-256,
the complete source-to-member mapping, every materialized split digest and
count, and all cross-split leakage invariants. Omitting `source_path` still
verifies the frozen materialized artifacts and their recorded membership.

Training code can call `load_frozen_dataset(manifest, "train", tokenizer)`.
Evaluation plumbing can call `iter_frozen_records(manifest, "held_out")`.
Both loaders verify and read recorded membership; neither computes a new split.

Tokenizer and serializer statistics are immutable sidecars produced with
`write_token_statistics`. Each sidecar pins:

- the split-manifest SHA-256 and all three source-level split digests;
- model and tokenizer identity plus their immutable revisions;
- serializer identity, version, and SHA-256; and
- per-split token and truncation counts.

Generating another model-specific sidecar never edits `split_manifest.json` or
any source-level split digest.

## Minimal paired-evaluation foundation

`agoge_forger.eval.contract` defines the versioned
`agoge.evaluation-contract.v1` schema. It consumes the exact held-out IDs and
digest from the frozen split manifest. Validation fails closed when causal base
and SFT arms drift in logical task identity, tokenizer provenance, serializer
identity/hash, decoding settings, context window, truncation policy, or scoring
version.

This foundation does not load a model, run inference, choose a checkpoint,
score generations, or create claim-bearing results. Those #100 capabilities
remain downstream work after the data contract is integrated.
