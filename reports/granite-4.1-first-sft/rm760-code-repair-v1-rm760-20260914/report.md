# RM-760 measured Granite 4.1 G0 → G1 (code-repair)

**Conclusion (pre-registered `compare.py` rules):** **improved**

Frozen contract: `experiment-contract.json`. Knobs were not changed after the committed freeze.

## G0 (base-only, before training)

- Bundle: `g0-eval/g0-base/`
- Held-out tasks: 16 scored, 0 invalid
- Base exact-match accuracy: **0.0000** (0/16)

## G1 (one bounded QLoRA epoch)

- Adapter (local, not in git): `/home/raulmc/agoge-data/adapters/granite-4.1-rm760-code-repair-v1-rm760-20260914/rm760-code-repair-v1-rm760-20260914`
- `adapter_model.safetensors` SHA-256: `3d96278d086c9738140c0bf040931824dd7aeea23bb50c778e49310e9bdcb558`
- Train: 129 rows, ~47 s wall-clock, final train loss **0.009792**, peak VRAM **5.57 GiB**

## Paired held-out eval

- Bundle: `g1-eval/` (base arm regenerated under the same decoding contract)
- Base accuracy: **0.0000**; SFT accuracy: **0.6250** (10/16)
- Paired: **10 improved**, **0 regressed**, **6 tied**, **0 invalid**
- Delta (SFT − base): **+0.6250**
- Scorer: `exact-match-v1`; held-out split SHA-256: `b5acdc7b4ebefce84ad3895aadcbf7ea4b3fe6e307fe80011e256d300ca18cb8`

## Notes

- G0 was run with `--device-map cuda:0` after stopping a CPU-offloaded run.
- Paired eval used Hub id `ibm-granite/granite-4.1-3b-base` (matches adapter `base_model_name_or_path`).
- Training `output_dir` with a literal `~` initially wrote the final adapter under `agoge-forger/~/agoge-data/…`; artifacts were consolidated under `~/agoge-data/…` before eval.
