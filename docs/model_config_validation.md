# Model Config Validation

Custom and experimental architectures (e.g., Nemotron, GKA-HQwen3) often use non-standard linear layers. Hardcoding `target_modules` like `["q_proj", "v_proj"]` can fail silently or lead to invalid training setups.

Agoge Forger prevents config hallucination through validation checks:

## Metadata Inspection
Before downloading large weights, you can inspect architecture details:
```bash
agoge model-metadata --model-id Qwen/Qwen2.5-1.5B-Instruct
```

## `target_modules_mode`
- `auto_common`: Automatically selects from standard Q/K/V/O/Gate/Up/Down projections if they exist.
- `discover_required`: Training will fail unless `target_modules` explicitly matches an inspected target report.
- `explicit`: Fails if a specified target module does not exist in the model graph.
