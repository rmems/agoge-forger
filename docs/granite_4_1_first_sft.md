# Granite 4.1 first measured SFT

One causal comparison:

```text
G0 — untouched ibm-granite/granite-4.1-3b-base
G1 — that pinned base plus one frozen code-repair SFT intervention
```

The research question, budget, decoding settings, and interpretation rule are
pre-registered in `agoge.granite-first-sft-contract.v1`. Positive, negative,
null, mixed, and inconclusive are all valid outcomes. Training loss is
diagnostic and cannot replace the held-out conclusion.

## Blockers

`freeze-granite-first-sft` refuses to write a contract until all of these hold:

- completion-only loss with `reject_over_budget` truncation, which is the
  training path from the completion-only trainer;
- a passing `agoge.model-compatibility.v1` report for
  `ibm-granite/granite-4.1-3b-base` whose lifecycle is
  `load-train-save-reload-generate`;
- the held-out harness scorer `exact-match-v1`;
- a frozen split manifest whose train, validation, and held-out digests are
  copied from the file rather than typed in.

The qualification report is the evidence that MiniCPM5-then-Granite
load/train/save/reload/generation passed. This command does not create that
report. A failed or missing verdict blocks the freeze.

```json
{
  "schema_version": "agoge.model-compatibility.v1",
  "model_repository": "ibm-granite/granite-4.1-3b-base",
  "model_revision": "<40-64 hex>",
  "tokenizer_repository": "ibm-granite/granite-4.1-3b-base",
  "tokenizer_revision": "<40-64 hex>",
  "tokenizer_sha256": "<64 hex>",
  "target_modules": ["q_proj", "v_proj"],
  "verdict": "pass",
  "lifecycle": "load-train-save-reload-generate",
  "trust_remote_code": false
}
```

`target_modules` must be the architecture-derived set recorded by that
qualification. The registered budget does not invent module names.

## Registered budget

Budget id `granite-4.1-first-sft-budget-v1`:

| Field | Value |
| --- | --- |
| Optimizer | `adamw_torch` |
| Learning rate | `0.0001` |
| Epochs | `1` |
| Batch size | `1` |
| Gradient accumulation | `8` (effective batch `8`) |
| Seed | `42` |
| Max sequence length | `2048`, refuse rows that do not fit |
| Loss | completion-only, Unicode code-point boundary |
| QLoRA | 4-bit NF4, bfloat16 compute, double quant |
| LoRA | `r=16`, `alpha=32`, dropout `0.05`, explicit targets |
| Decoding | greedy, seed `17`, `max_new_tokens=128` |
| Context window | `2048`, `mark_unsupported` when the prompt does not fit |
| Primary metric | held-out `exact-match-v1` accuracy |
| Protected metric | none |

Changing any of these values requires a new budget id and a new experiment.
Do not retune them after seeing G0.

## Commands

Run these on the local CUDA machine after the qualification report and the
frozen code-repair split exist. The worktree must be clean.

```bash
uv run agoge freeze-granite-first-sft \
  --experiment-id granite-4-1-first \
  --output-dir reports/granite-4.1-first-sft/granite-4-1-first \
  --split-manifest /path/to/split_manifest.json \
  --qualification-report /path/to/granite-compatibility.json

uv run agoge run-granite-first-sft \
  --contract reports/granite-4.1-first-sft/granite-4-1-first/experiment-contract.json
```

`run-granite-first-sft` publishes G0 before it calls training. G1 scoring runs
only after training completes and a new interpreter reloads the adapter. A
failed stage leaves `comparison.json` with `"status": "not_run"` and a report
that does not invent a quality conclusion.

## Artifacts

```text
reports/granite-4.1-first-sft/<experiment_id>/
  experiment-contract.json
  experiment-contract.sha256
  g0-base/
  g1-sft/
  comparison.json
  regressions.jsonl
  training-results.json
  report.md
  reproduction.md
```

Accepted tokens are the shifted supervised completion tokens recorded in
`completion_preprocessing.json`. The adapter is `g1-sft/adapter`.

## What CI proves

CI freezes a contract against a fixture split and a fixture qualification
file, then runs the protocol with injected generations. That proves G0 is
published before training, that a G0 or training failure does not emit a
paired conclusion, and that the written conclusion matches `compare_arms`.
It does not load Granite weights and it is not a measured result.
