# Run Manifest Schema

Every training or inference run produces a `manifest.json` that captures the full context of the run for reproducibility.

## File Location

```
runs/<run_name>/manifest.json
```

## Schema

```json
{
  "timestamp": "2025-01-15T14:30:00Z",
  "git": {
    "commit": "a1b2c3d",
    "branch": "main",
    "dirty": false
  },
  "config": {},
  "metrics": {
    "max_vram_gb": 12.5,
    "gpu_report": {
      "device_name": "NVIDIA RTX 5080",
      "compute_capability": [10, 1],
      "total_vram_gb": 16.0,
      "allocated_vram_gb": 12.5,
      "bf16_supported": true
    },
    "artifact_index": "adapters/my-run/artifact_index.json"
  },
  "environment": {
    "python_version": "3.12.0",
    "torch_version": "2.4.0",
    "cuda_version": "12.4"
  },
  "model_metadata": {
    "dtype": "torch.bfloat16",
    "parameters": 8030261248
  },
  "tokenizer": {
    "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
    "vocab_size": 128256
  },
  "dataset": {
    "num_rows": 1000
  }
}
```

## Fields

| Field             | Type   | Required | Description                                        |
|-------------------|--------|----------|----------------------------------------------------|
| `timestamp`       | str    | Yes      | ISO 8601 UTC datetime of run start                 |
| `git`             | object | Yes      | Git state at run time                              |
| `git.commit`      | str    | Yes      | Full or short commit SHA                           |
| `git.branch`      | str    | Yes      | Branch name                                        |
| `git.dirty`       | bool   | Yes      | Whether uncommitted changes were present           |
| `config`          | object | Yes      | Full `ExperimentConfig` (see config schema)        |
| `metrics`         | object | Yes      | Runtime metrics                                    |
| `metrics.max_vram_gb` | float | Yes  | Peak VRAM usage in GB                              |
| `metrics.gpu_report`  | object | Yes  | GPU hardware report (see below)                |
| `metrics.artifact_index` | str | Yes | Path to `artifact_index.json`                  |
| `environment`     | object | Yes      | Software environment                               |
| `environment.python_version` | str | Yes | Python version string                      |
| `environment.torch_version` | str | Yes | PyTorch version string                     |
| `environment.cuda_version` | str | Yes | CUDA version or `"None"`                   |
| `model_metadata`  | object | No       | Present if model is loaded                         |
| `model_metadata.dtype` | str | No      | Model dtype (e.g. `"torch.bfloat16"`)            |
| `model_metadata.parameters` | int | No   | Total parameter count                              |
| `tokenizer`       | object | No       | Present if tokenizer is loaded                     |
| `tokenizer.name_or_path` | str | No    | Tokenizer name or path                             |
| `tokenizer.vocab_size` | int | No      | Vocabulary size                                    |
| `dataset`         | object | No       | Present if dataset is loaded                       |
| `dataset.num_rows` | int   | No       | Number of rows in the dataset                      |

### GPU Report Object

| Field                    | Type         | Description                      |
|--------------------------|--------------|----------------------------------|
| `device_name`            | str          | GPU device name                  |
| `compute_capability`     | list[int]    | Major, minor compute capability  |
| `total_vram_gb`          | float        | Total VRAM in GB                 |
| `allocated_vram_gb`      | float        | Allocated VRAM in GB             |
| `bf16_supported`         | bool         | Whether BF16 is supported        |

## Owner

| Language | Responsibility                                            |
|----------|-----------------------------------------------------------|
| Python   | Writes `manifest.json` during training and eval runs      |
| Rust     | Reads manifest for artifact discovery and validation      |
| Julia    | Reads manifest for run context (model, dataset info)      |

## Notes

- `config` contains the full `ExperimentConfig.model_dump()` output
- All JSON output uses `indent=2` formatting
- `cuda_version` is `"None"` (string) when CUDA is unavailable
