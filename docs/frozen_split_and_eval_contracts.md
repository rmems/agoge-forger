# Frozen split and evaluation contracts

Agoge freezes source-level task identity before any model-specific rendering or
tokenization. `SplitManifest` in `agoge_forger.split_contract` is the one
authoritative runtime schema. Its version is `agoge.split-manifest.v1`.

## Curated source requirements

The input is a local JSONL file from an explicitly named repository, immutable
revision, dataset version, and canonical repository-relative source path. The
local file may be a cache with a different basename; the recorded source path
is supplied explicitly and is never inferred from that local filename. Every
source record must contain:

- `canonical_id`: globally stable sample identity;
- `lineage_id`: identity shared by variants that must remain together; and
- training content accepted by Agoge, such as `text`, `messages`, or
  `instruction`/`output`.

`group_id` is optional. When present, the complete group remains atomic. Exact
canonical content, lineage, and declared-group relationships are transitively
joined before split assignment. Identity field names can be overridden for a
versioned source that already has equivalent fields.

A frozen source must use exactly one training-content representation: `text`,
`messages`, or `instruction`/`output`. Agoge rejects mixed representations
because a tokenizer-specific chat template can make a `messages` row equivalent
to a `text` row even though model-independent fallback rendering cannot prove
that equivalence. Keeping one representation per snapshot makes canonical split
membership independent of the eventual model, tokenizer, and serializer.

## Materialize one frozen snapshot

From an installed checkout:

```bash
uv run python scripts/freeze_split.py \
  --source /path/to/curated.jsonl \
  --source-path datasets/curated/sft.jsonl \
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

`--source-path` must be a canonical, confined POSIX-relative path such as
`datasets/curated/sft.jsonl`; absolute paths, drive-prefixed paths, backslashes,
and `.` or `..` traversal are rejected. The report states that exact source
path, source coverage, record counts, partition digests,
deterministic leakage guarantees, exclusions, and limitations. Semantic
near-duplicate detection is deliberately not claimed by the deterministic
gate.

## Validate and consume without re-splitting

`validate_split_manifest(path, source_path=...)` verifies the source SHA-256,
the complete source-to-member mapping (using the recorded repository-relative
path for source coordinates), every materialized split digest and count, and all
cross-split leakage invariants. Omitting `source_path` still verifies the frozen
materialized artifacts and their recorded membership.

Training code can call `load_frozen_dataset(manifest, "train", tokenizer)`.
Evaluation plumbing can call `iter_frozen_records(manifest, "held_out")`.
Both loaders verify and read recorded membership; neither computes a new split.

Tokenizer and serializer statistics are immutable sidecars produced with
`write_token_statistics`. Callers must supply `TokenizerBinding` and
`SerializerBinding`; the writer rejects any mismatch between the declared spec
and the bound callable provenance before writing output. Each sidecar pins:

- the split-manifest SHA-256 and all three source-level split digests;
- model identity and immutable revision, plus tokenizer identity, immutable
  revision, and a SHA-256 derived from the bound implementation and canonical
  tokenizer state;
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
