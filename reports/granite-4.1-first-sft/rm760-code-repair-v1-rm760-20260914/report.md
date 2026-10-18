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

## Secondary scoring (post hoc, not pre-registered)

The conclusion above stands as the pre-registered result. After it was recorded, #183 found that
`exact-match-v1` compares raw text, and the base model wrapped **16/16** outputs in a markdown code
fence while the SFT model wrapped **0/16**. This section re-scores the *same committed*
`g1-eval/{base,sft}/generations.jsonl` (nothing regenerated, `experiment-contract.json` untouched) with
two scorers that are not named in the contract. They were chosen after seeing the v1 result, so treat
them as a diagnostic of the confound, not as a replacement result.

| Scorer | Base correct | SFT correct | Delta (SFT − base) | Improved / regressed / tied / invalid | `compare.py` label |
| --- | ---: | ---: | ---: | ---: | --- |
| `exact-match-v1` (pre-registered) | 0/16 | 10/16 | +0.6250 | 10 / 0 / 6 / 0 | improved |
| `exact-match-v2` (leading fence unwrapped) | 10/16 | 10/16 | +0.0000 | 2 / 2 / 12 / 0 | mixed |
| `doctest-v1` (run the module's doctests) | 11/16 | 10/16 | −0.0625 | 2 / 3 / 11 / 0 | mixed |

- The whole v1 delta is fence formatting. With the wrapper removed, base and SFT both get 10/16 exact
  matches; the SFT's fence-free formatting, not better repairs, is what v1 rewarded.
- Per-task differences under v2: SFT better on `…-00132` and `…-00211`, base better on `…-00025` and
  `…-00042`. `doctest-v1` agrees and also scores base `…-00128` correct (a different fix that passes
  all five doctests) where SFT fails them.
- Seven of 16 base replies keep going after the code block (an explanation, sometimes a second block).
  Both new scorers score the **first** fenced block and ignore what follows. A stricter reading that
  only accepts a reply that is exactly one fenced block would give base 6/16 under v2; SFT is 10/16 under
  either reading, so that choice changes the size of the gap but not that v1's 0/16 was an artefact.
- 16 held-out tasks cannot separate a 1-3 task difference from noise; none of these deltas is significant.

Files: `comparison.exact-match-v2.json`, `comparison.doctest-v1.json` (per-task verdicts, plus
`n_examples` / `n_failed` / `error` for doctest). Regenerate with:

```bash
for v in exact-match-v2 doctest-v1; do
  uv run agoge rescore-held-out reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/g1-eval \
    --scoring-version "$v" \
    --output reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/comparison."$v".json
done
```
