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
- All JSON output uses `indent=2` formatting
