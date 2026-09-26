# Granite 4.1 G0 → G1 on R2EGym (`r2egym-cap512-last-assistant-v2-chunked8192`)

**Conclusion (pre-registered `exact-match-v1`):** **null**

New experiment after v1 8192 QLoRA OOM. Same freeze. Trainer uses TRL `chunked_nll` and activation offload (same NLL, no full-sequence `lm_head` logits). Knobs were not changed after seeing this result.

## G0 (base-only, before training)

- Copied from v1 (`g0-eval/`) because eval knobs, split, and base revision match; paired eval also regenerated the base arm under `g1-eval/base/`.
- Held-out: 55 tasks; **24 scored**, **31 invalid** (`mark_unsupported`)
- Base exact-match: **0.0000** (0/24)

## G1 (one bounded QLoRA epoch)

- Adapter (local): `/home/raulmc/agoge-data/adapters/granite-4.1-r2egym-cap512-last-assistant-v2-chunked8192/r2egym-cap512-last-assistant-v2-chunked8192`
- `adapter_model.safetensors` SHA-256: `bc89d1befeb26bd3f5cfee8d484d942190c5330ec7893911576333555eb2820a`
- Train: 121/411 rows (290 over 8192 dropped), ~413 s, train loss **3.29**, peak VRAM **7.30 GiB**
- `loss_type: chunked_nll`, `activation_offloading: true`

## Paired held-out eval

- Bundle: `g1-eval/`
- Base **0/24**; SFT **0/24**; Δ **0**; 0 improved, 0 regressed, 24 tied, 31 invalid
- Scorer: `exact-match-v1` on the last `Assistant: ` turn including the header. That is a hard copy of a high-entropy wrap-up; 0/24 is the expected floor, not a claim that the adapter did nothing.

## Notes

- Do not use 4096 on this freeze: only 1 train row and 1 held-out prompt fit.
- Unload Ollama before GPU jobs on this host.
