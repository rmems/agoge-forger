# Metrics Schema

Metrics are currently embedded within `manifest.json` under the `metrics` key. There is no standalone CSV or Parquet output yet. This document describes the current metrics structure and the planned CSV/Parquet format for future use.

## Current Metrics (Embedded in Manifest)

```json
{
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
  }
}
```

### Fields

| Field                     | Type   | Description                             |
|---------------------------|--------|-----------------------------------------|
| `max_vram_gb`             | float  | Peak VRAM usage in GB                   |
| `gpu_report`              | object | GPU hardware and capability details     |
| `gpu_report.device_name`  | str    | GPU device name                         |
| `gpu_report.compute_capability` | list[int] | Compute capability (major, minor) |
| `gpu_report.total_vram_gb` | float | Total VRAM available in GB             |
| `gpu_report.allocated_vram_gb` | float | VRAM allocated at measurement time |
| `gpu_report.bf16_supported` | bool  | Whether BF16 is supported             |
| `artifact_index`          | str    | Path to artifact_index.json             |

## Planned CSV/Parquet Format

For cross-language metrics aggregation (Python training results + Julia statistical analysis), the following tabular format is planned:

### File Location

```
runs/<run_name>/metrics.csv       # CSV format
runs/<run_name>/metrics.parquet   # Parquet format (if pyarrow available)
```

### Column Schema

| Column                | Type   | Description                                |
|-----------------------|--------|--------------------------------------------|
| `run_name`            | str    | Run identifier                             |
| `timestamp`           | str    | ISO 8601 UTC timestamp                     |
| `model_id`            | str    | Model identifier                           |
| `dataset_path`        | str    | Dataset path                               |
| `max_vram_gb`         | float  | Peak VRAM usage                            |
| `total_vram_gb`       | float  | Total available VRAM                       |
| `device_name`         | str    | GPU device name                            |
| `bf16_supported`      | bool   | BF16 support flag                          |
| `python_version`      | str    | Python version                             |
| `torch_version`       | str    | PyTorch version                            |
| `cuda_version`        | str    | CUDA version                               |
| `quantization`        | str    | Quantization mode (e.g. "4bit_nf4")        |
| `lora_r`              | int    | LoRA rank                                  |
| `lora_alpha`          | int    | LoRA alpha                                 |
| `learning_rate`       | float  | Learning rate                              |
| `batch_size`          | int    | Per-device batch size                      |
| `num_train_epochs`    | int    | Number of training epochs                  |
| `max_seq_length`      | int    | Maximum sequence length                    |
| `final_loss`          | float  | Final training loss (if available)         |
| `training_time_s`     | float  | Wall-clock training time in seconds        |
| `num_parameters`      | int    | Total model parameters                     |
| `adapter_size_bytes`  | int    | Total adapter artifact size                |

### Usage

```python
import pandas as pd

df = pd.read_csv("runs/my-run/metrics.csv")
# or
df = pd.read_parquet("runs/my-run/metrics.parquet")
```

## Owner

| Language | Responsibility                                           |
|----------|----------------------------------------------------------|
| Python   | Writes metrics (current: manifest; planned: CSV/Parquet) |
| Julia    | Reads CSV/Parquet for statistical analysis               |
| Rust     | Reads CSV for benchmark comparison                       |

## Notes

- CSV and Parquet outputs are **not yet implemented** — this schema is forward-looking
- Parquet requires `pyarrow` as an optional dependency
- When implemented, both CSV and Parquet will contain the same columns
- CSV is the primary format for maximum compatibility; Parquet is optional for large-scale analysis
