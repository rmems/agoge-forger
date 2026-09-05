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
- pre-rendered `text`: the exact model-independent training content.

`group_id` is optional. When present, the complete group remains atomic. Exact
canonical content, lineage, and declared-group relationships are transitively
joined before split assignment. Identity field names can be overridden for a
versioned source that already has equivalent fields.

A new frozen source must use pre-rendered `text`. Agoge rejects `messages` and
`instruction`/`output` rows because downstream tokenizer templates and
serializers can render them differently. Binding the exact rendered text before
split assignment keeps canonical content identity and split membership
independent of the eventual model, tokenizer, and serializer.

## Materialize one frozen snapshot

From an installed checkout:

```bash
uv run agoge freeze-split \
  --source /path/to/curated.jsonl \
  --source-path datasets/curated/sft.jsonl \
  --output-dir /path/to/new-snapshot \
  --source-repository rmems/synthetic-factory \
  --source-revision <immutable-commit> \
  --dataset-version <version> \
  --seed 20260830 \
  --salt <versioned-policy-salt>
```

Or call the library directly:

```python
from agoge_forger.split_contract import (
    SplitMaterializationSpec,
    SplitPolicy,
    materialize_split,
)

manifest = materialize_split(
    "/path/to/curated.jsonl",
    "/path/to/new-snapshot",
    SplitMaterializationSpec(
        source_repository="rmems/synthetic-factory",
        source_revision="<immutable-commit>",
        dataset_version="<version>",
        source_path="datasets/curated/sft.jsonl",
        split_policy=SplitPolicy(
            seed=20260830,
            salt="<versioned-policy-salt>",
            weights={"train": 80, "validation": 10, "held_out": 10},
        ),
    ),
)
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

Snapshot materialization currently requires Linux `renameat2` support for
atomic no-replace directory publication. The capability is checked before any
source or payload staging begins. Validation on platforms without safe
descriptor-relative traversal fails with a controlled diagnostic.

Validation normally creates short-lived verified copies beside the manifest,
optional source, and model artifact bundle so they remain on the same
filesystem. For read-only snapshot or artifact mounts, set
`AGOGE_VALIDATION_STAGING_DIR` to an existing writable directory. Validation
removes those temporary copies after use; it never edits the frozen inputs.
When validating an adapter or merged-model bundle, provision staging capacity
for all files named by its artifact index.

`validate_split_manifest(path, source_path=...)` verifies the source SHA-256,
the complete source-to-member mapping (using the recorded repository-relative
path for source coordinates), every materialized split digest and count, and all
cross-split leakage invariants. Omitting `source_path` still verifies the frozen
materialized artifacts and their recorded membership.

Training code can call
`load_frozen_dataset(manifest_path, "train", tokenizer)`, where `manifest_path`
is `/path/to/new-snapshot/split_manifest.json`. Evaluation plumbing can call
`iter_frozen_records(manifest_path, "held_out")`.
Both loaders verify and read recorded membership; neither computes a new split.
The dataset loader includes the exact manifest and selected-split digests in
the Hugging Face generator cache identity, so replacing files at the same path
cannot silently reuse stale Arrow membership.

Trainer and training-config wiring for frozen splits is **not shipped in this
PR**. `ExperimentConfig` still requires `dataset_path`; there is no
`split_manifest_path` / `split_name` training config surface yet.
CLI train and export paths construct `producer_provenance` (base model, immutable
revision, `training_split_manifest_sha256`, `training_split_name="train"`, and
`training_split_sha256`) from `config.model_id`, a content-addressed
`config.revision`, and freeze metadata beside `dataset_path`, or fail closed
before save. `write_artifact_index` still emits that object only when a caller
supplies it. Callers may use the freeze/validate/loader APIs above directly;
pinning a manifest into `ExperimentConfig` remains follow-on work.

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
`agoge.evaluation-contract.v2` schema. It consumes the exact held-out IDs and
digest from the frozen split manifest. Validation fails closed when causal base
and SFT arms drift in logical task identity, tokenizer repository, revision, or
canonical-state SHA-256, serializer identity/hash, decoding settings, context
window, truncation policy, or scoring version. Version 1 contracts do not carry
the tokenizer-state digest and must be rebuilt as version 2 before measured
evaluation. Artifact validation additionally requires both adapter and merged SFT
bundles to prove they were produced from that exact manifest's `train`
artifact. Contract-relative paths use POSIX separators for relocation between
Windows and Unix systems, and absolute or drive-prefixed references are
rejected during schema validation.

When an artifact index carries `producer_provenance`, paired-eval validation
requires that provenance to match the contract's frozen train split. CLI train
and export derive that object from the pinned model revision and frozen train
split beside `dataset_path` (or from an adapter index on merge) and fail closed
when those digests are missing. Trainer-side settings such as
`trust_remote_code: false` and `runtime.save_safetensors: true` for
evaluation-eligible frozen runs are likewise not wired through config in this
PR. Legacy `dataset_path` training keeps its existing explicit
unsafe-serialization opt-in.

This foundation does not load a model, run inference, choose a checkpoint,
score generations, create claim-bearing results, or wire frozen splits into
the trainer. Those capabilities remain downstream work after the data
contract is integrated.
