# Artifact Index Schema

Every output directory that contains model artifacts produces an `artifact_index.json` that lists all files, their sizes, and SHA-256 checksums for integrity verification.

## File Location

```
adapters/<run_name>/artifact_index.json   # training output
<merge_output_dir>/artifact_index.json    # merge output
```

## Schema

```json
{
  "output_dir": "/path/to/output",
  "producer_provenance": {
    "base_model_name_or_path": "ibm-granite/granite-4.1-3b-base",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "training_split_manifest_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "training_split_name": "train",
    "training_split_sha256": "2222222222222222222222222222222222222222222222222222222222222222"
  },
  "artifacts": [
    {
      "file": "adapter_model.safetensors",
      "size_bytes": 134217728,
      "sha256": "e3b0c44298fc1c149afbf4c8996fb924..."
    },
    {
      "file": "adapter_config.json",
      "size_bytes": 842,
      "sha256": "a1b2c3d4e5f6..."
    }
  ]
}
```

## Fields

| Field               | Type   | Required | Description                                    |
|---------------------|--------|----------|------------------------------------------------|
| `output_dir`        | str    | Yes      | Path to the output directory (as written by producer — may be relative or absolute) |
| `artifacts`         | list   | Yes      | Array of artifact objects                      |
| `artifacts[].file`  | str    | Yes      | Relative path from the directory containing `artifact_index.json` (normally `output_dir`) |
| `artifacts[].size_bytes` | int | Yes    | File size in bytes                             |
| `artifacts[].sha256` | str   | Yes      | Hex-encoded SHA-256 hash of the file contents  |
| `producer_provenance` | object | No | Immutable model and frozen-training identity emitted by an eligible producer |

Evaluation-eligible adapter and merged-model bundles require all five
`producer_provenance` fields shown above. Legacy JSONL training remains
supported, but its indexes omit this object and cannot be used as the SFT arm
of a paired evaluation contract. Merging a provenanced adapter copies its
validated provenance unchanged; merge code never invents training provenance.
Default merge requires that verified index. `allow_unsafe=True` is the explicit
legacy escape hatch and produces an unprovenanced, evaluation-ineligible merge.

## Referenced By

The `manifest.json` `metrics.artifact_index` field contains the path to the artifact index for the run.

## Typical Artifacts

### Training Output (`adapters/<run_name>/`)

| File                           | Description                          |
|--------------------------------|--------------------------------------|
| `adapter_model.safetensors`   | LoRA adapter weights                 |
| `adapter_config.json`          | PEFT adapter configuration           |
| `special_tokens_map.json`      | Tokenizer special tokens             |
| `tokenizer_config.json`        | Tokenizer configuration              |
| `tokenizer.json`               | Fast tokenizer data                  |
| `trainer_state.json`           | Trainer state with loss history      |
| `training_args.bin`            | Serialized training arguments        |
| `artifact_index.json`          | This index file                      |

### Merge Output (`<merge_dir>/`)

| File                           | Description                          |
|--------------------------------|--------------------------------------|
| `model.safetensors`            | Full merged model weights (sharded)  |
| `model.safetensors.index.json` | Shard index for multi-file models    |
| `config.json`                  | Model configuration                  |
| `generation_config.json`       | Generation configuration             |
| `artifact_index.json`          | This index file                      |

## Owner

Python writes and consumes `artifact_index.json` after training and merging.

## Notes

- `sha256` is computed on the raw file bytes before any encoding
- `file` paths use the platform's native path separator (backslashes on Windows, forward slashes on Unix)
- The index file itself is not listed in the `artifacts` array
- Consumers resolve listed files from the directory containing `artifact_index.json`; `output_dir` is producer provenance and may become stale when an immutable artifact bundle is relocated
- Consumers verify every listed file's existence, regular-file type, byte size, and SHA-256 digest before using the bundle
- Evaluation-contract validation also requires the index to cover every regular file recursively beneath the bundle root, excluding only the root `artifact_index.json`, and rejects symlinks or special files
- A `peft_adapter` evaluation artifact requires `adapter_config.json` plus `adapter_model.safetensors` and binds the adapter's base repository, immutable revision, exact split manifest, and frozen `train` digest to the SFT arm; a `merged_model` requires the same producer provenance plus `config.json` and either one `model.safetensors` or a complete safetensors shard index
- All JSON output uses `indent=2` formatting
