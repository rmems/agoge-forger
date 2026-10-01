# Granite 4.1 G0 → G1 on R2EGym (`r2egym-cap512-last-assistant-v2-seq4096`)

**Conclusion:** **Not run.** After token counts, only **1/411** train rows and **1/55** held-out prompts fit 4096, so this budget is not a measured G0/G1. The live follow-up is `r2egym-cap512-last-assistant-v2-chunked8192` (same freeze, 8192, TRL `chunked_nll`).

Frozen contract: `experiment-contract.json`. Split membership is the copied v1 `pinned-split/` (same digests; not re-split).

## G0 (base-only, before training)

Not run.

## G1 (one bounded QLoRA epoch)

Not run.

## Paired held-out eval

Not run. `agoge held-out-eval` requires an SFT artifact.

## Notes

- Last-assistant `completion_start_char` is the line-start of the final `Assistant: ` header. Exact-match includes that header.
- Unload Ollama embedding models before GPU jobs on this machine.
