# Model compatibility qualify — 2026-09-11

**Issue:** agoge-forger #104 / Linear RM-779  
**Machine:** ShipOfTheseus (RTX 5080)  
**Agoge code used:** `6d779e9f` (main incl. #138; worktree `~/agoge-data/scratch/wt-main-qualify`)  
**Claim level:** compatibility / plumbing only — **not** measured SFT (#101).

## Gate status

| Gate | Status |
| --- | --- |
| #133 / RM-1032 completion-only on real trainer batch | **DONE** (PR #138 / `6d779e9f`) |
| MiniCPM5 load→train→save→clean reload→generation | **PASS** (mechanical; see caveats) |
| Granite 4.1 load→train→save | **PASS** (tiny + R2EGym cap-512) |
| Granite 4.1 clean reload→generation | **PASS** (`smoke-eval` on tiny adapter) |
| Formal bundle under `~/agoge-data/reports/model-compatibility/2026-09-11/` | **UPDATED** |

## MiniCPM5 canary

- Revision: `156170697656c48f69915b33a2fb44110242187c`
- Targets: q/k/v/o/gate/up/down_proj (inspected + frozen in config)
- Split: `tiny-sft-smoke-v1` train 72 rows, `max_seq_length` 512
- Adapter: `~/agoge-data/scratch/smoke/adapters/local_minicpm5_canary`
- adapter sha256: `af36e2b0893fd83f3b8bd568d4883b61b4de54b230be8683259ff5bb11e3a29b`
- train_loss 2.683, runtime 8.6s, max VRAM 2.49 GiB, 18 steps
- `smoke-eval` EXIT 0 (clean reload + 3 prompts)
- **Caveat:** `completion_only_loss: false` because this tiny chat freeze has no `completion_start_char`. A derived canary freeze with that field failed membership revalidation (canonical encoding). Code-repair completion-masked MiniCPM run remains optional; #138 already covers the trainer-batch contract.

## Granite 4.1

### Tiny plumbing
- Revision: `dacb9cb9157bec98e99b09f285c92a4d58405c96`
- Adapter sha256: `c3cc0a328442a5d10d33f33d8199a010a73e3d610952054428cb60bb897f0eaf`
- train_loss ≈1.046, 14.5s, 3.38 GiB, 36 steps
- `smoke-eval` EXIT 0

### R2EGym cap-512
- Adapter sha256: `d769822d864d5ca145d101f9b79e0c7587fdf6299ebada56f3d2057f0bc9aeb6`
- train_loss 0.3726, 468.4s, 6.36 GiB, 102 steps
- (generation smoke recorded on tiny adapter; R2EGym reload not re-run tonight)

## Suggested close posture

Mechanical #104 lifecycle is green for both models on ShipOfTheseus. Remaining honesty gap is an optional MiniCPM **code-repair** completion-masked canary once a frozen repair export with `completion_start_char` (Agoge canonical JSONL) exists. Default recommendation: treat mechanical qualify as sufficient to unlock #100 harness work unless Raul wants that optional canary first.
