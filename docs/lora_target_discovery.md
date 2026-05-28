# LoRA Target Discovery

You should not guess target modules for unknown models. Use the discovery tool to analyze the module graph and emit a JSON report:

```bash
agoge inspect-lora-targets --model-id amazon/GKA-primed-HQwen3-8B-Reasoner --out runs/inspect/gka_targets.json
```

This will group layers by leaf names, showing their parameter counts and base types (e.g., `torch.nn.Linear`, `Linear4bit`). 

Custom architectures often mix standard dense layers with state-space models (SSMs), Kalman filters, or grouped-query specific attention projections. Always inspect these before adding them to your LoRA config.
