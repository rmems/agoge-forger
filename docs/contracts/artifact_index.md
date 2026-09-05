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
  ],
  "producer_provenance": {
    "base_model_name_or_path": "org/model",
    "revision": "abcdef0123456789abcdef0123456789abcdef01",
    "training_split_manifest_sha256": "7f...64 hex chars...",
    "training_split_name": "train",
    "training_split_sha256": "8f...64 hex chars..."
  }
}
```

`producer_provenance` is optional on the writer. `write_artifact_index` emits it
only when the caller supplies a valid `ArtifactProducerProvenance` object or
mapping. CLI train and export paths always construct and pass that object from
the pinned base-model revision and frozen train-split digests (located beside
`dataset_path` or copied from an adapter index) or fail closed before save.
`ExperimentConfig` does not have a `split_manifest_path` field.

## Fields

| Field               | Type   | Required | Description                                    |
|---------------------|--------|----------|------------------------------------------------|
| `output_dir`        | str    | Yes      | Path to the output directory (as written by producer — may be relative or absolute) |
| `artifacts`         | list   | Yes      | Array of artifact objects                      |
| `artifacts[].file`  | str    | Yes      | Relative path from `output_dir`                |
| `artifacts[].size_bytes` | int | Yes    | File size in bytes                             |
| `artifacts[].sha256` | str   | Yes      | Hex-encoded SHA-256 hash of the file contents  |
| `producer_provenance` | object | No     | Present only when a caller supplies training identity |
| `producer_provenance.base_model_name_or_path` | str | Yes, if present | Base model repository or path |
| `producer_provenance.revision` | str | Yes, if present | Content-addressed revision (`^[0-9a-f]{40,64}$`) |
| `producer_provenance.training_split_manifest_sha256` | str | Yes, if present | SHA-256 of the frozen split manifest |
| `producer_provenance.training_split_name` | str | Yes, if present | Must be `"train"` |
| `producer_provenance.training_split_sha256` | str | Yes, if present | SHA-256 of the frozen train split |

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
- All JSON output uses `indent=2` formatting
