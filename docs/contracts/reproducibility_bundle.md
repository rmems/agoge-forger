# Reproducibility Bundle Schema

A reproducibility bundle is a local directory that binds one experiment's run
manifest, locked training config, frozen split identity, adapter and optional
merged artifact indexes, and paired evaluation contract. Verification is
offline: it hashes files and parses JSON contracts. It does not load model
weights, execute model code, or contact the network.

## File Location

```text
<bundle>/
  reproducibility-bundle.json
  run/manifest.json
  config/locked.json
  split_manifest.json
  splits/train.jsonl
  splits/validation.jsonl
  splits/held_out.jsonl
  adapter/artifact_index.json
  adapter/adapter_config.json
  adapter/adapter_model.safetensors
  contract.json
```

Merged artifacts are optional. When present they live beside a second
`artifact_index.json` and are named from `merged_artifact_index_path`.

## Schema

`reproducibility-bundle.json` is `agoge.reproducibility-bundle.v1`. It lists
every file in the bundle except itself.

```json
{
  "schema_version": "agoge.reproducibility-bundle.v1",
  "run_manifest_path": "run/manifest.json",
  "locked_config_path": "config/locked.json",
  "split_manifest_path": "split_manifest.json",
  "adapter_artifact_index_path": "adapter/artifact_index.json",
  "merged_artifact_index_path": null,
  "evaluation_contract_path": "contract.json",
  "files": [
    {
      "path": "adapter/adapter_config.json",
      "size_bytes": 128,
      "sha256": "0123...64 hex chars..."
    }
  ]
}
```

Paths are canonical POSIX paths relative to the bundle root. Absolute paths,
drive prefixes, backslashes, `.` / `..` traversal, and a self-listing of
`reproducibility-bundle.json` are rejected. Duplicate inventory paths fail
closed.

## Verification

```bash
agoge verify-bundle /path/to/bundle
agoge verify-bundle /path/to/bundle --format table
```

`--format json` (default) prints an `agoge.bundle-verification.v1` verdict.
`--format table` prints a concise human report. Exit `0` only when the verdict
is `pass`. The verifier:

- refuses symlinks and descriptor-unsafe platforms;
- compares inventory membership against the tree (missing, extra, modified);
- validates run-manifest fields, locked `ExperimentConfig`, frozen split
  identity, adapter/merged artifact indexes, and both evaluation arms;
- fails clearly on unknown schema versions and truncated JSON.

It never deserializes safetensors tensors or pickle weight files.

## Owner

Python writes the inventory through `write_reproducibility_bundle` and consumes
it through `verify_reproducibility_bundle` / `agoge verify-bundle`.
