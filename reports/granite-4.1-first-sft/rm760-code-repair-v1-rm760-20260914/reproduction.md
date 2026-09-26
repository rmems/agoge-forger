# RM-760 reproduction (`rm760-code-repair-v1-rm760-20260914`)

Knobs are frozen in `experiment-contract.json`. Do not change them after G0.

## Dataset and split

```bash
# synthetic-factory @ dcc8cc0b7757064d78d0fa445f1d787f8ab99cfc
cd /home/raulmc/rmems/synthetic-factory
python3 pipelines/code_repair_cli.py catalog-check --catalog catalogs/python-repair-v1 --json
python3 pipelines/code_repair_cli.py generate --catalog catalogs/python-repair-v1 --seed 20260908 \
  --count 240 --per-program-cap 3 --produced-at 2026-09-14T00:00:00.000Z \
  --out outputs/code-repair/rm760-20260914 --json
python3 pipelines/code_repair_cli.py replay --run outputs/code-repair/rm760-20260914 \
  --catalog catalogs/python-repair-v1 --out outputs/code-repair/rm760-20260914-replay --json
python3 pipelines/code_repair_cli.py export --run outputs/code-repair/rm760-20260914 \
  --catalog catalogs/python-repair-v1 \
  --replay outputs/code-repair/rm760-20260914-replay \
  --out outputs/code-repair/rm760-20260914-export --json

cd /home/raulmc/rmems/agoge-forger
uv run agoge freeze-split \
  --source /home/raulmc/rmems/synthetic-factory/outputs/code-repair/rm760-20260914-export/agoge/code_repair_v1.jsonl \
  --source-path outputs/code-repair/rm760-20260914-export/agoge/code_repair_v1.jsonl \
  --output-dir /home/raulmc/agoge-data/splits/code-repair-v1-rm760-20260914 \
  --source-repository rmems/synthetic-factory \
  --source-revision dcc8cc0b7757064d78d0fa445f1d787f8ab99cfc \
  --dataset-version code-repair-v1-rm760-20260914 \
  --seed 20260908 --salt python-repair-v1
```

Pinned split copy for this experiment lives at `pinned-split/` next to this file.

## G0 (base only, before G1)

Unload conflicting GPU services (e.g. Ollama embeddings) before bf16 eval.

```bash
cd /home/raulmc/rmems/agoge-forger
uv run agoge g0-held-out-eval \
  --split-manifest reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/pinned-split/split_manifest.json \
  --output-dir reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/g0-eval \
  --experiment-id rm760-code-repair-v1-rm760-20260914 \
  --base-model-id /home/raulmc/.models/ibm-granite/granite-4.1-3b-base \
  --base-revision dacb9cb9157bec98e99b09f285c92a4d58405c96 \
  --context-window 832 --max-new-tokens 576 --seed 17 \
  --truncation-policy mark_unsupported \
  --device-map cuda:0
```

## G1 train

```bash
cd /home/raulmc/rmems/agoge-forger
uv run agoge train-qlora --config reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/train.yaml
```

## G1 paired held-out eval (after adapter index exists)

```bash
uv run agoge held-out-eval \
  --split-manifest reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/pinned-split/split_manifest.json \
  --sft-artifact /home/raulmc/agoge-data/adapters/granite-4.1-rm760-code-repair-v1-rm760-20260914/rm760-code-repair-v1-rm760-20260914 \
  --output-dir reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/g1-eval \
  --base-model-id ibm-granite/granite-4.1-3b-base \
  --base-revision dacb9cb9157bec98e99b09f285c92a4d58405c96 \
  --context-window 832 --max-new-tokens 576 --seed 17 \
  --truncation-policy mark_unsupported \
  --device-map cuda:0
```

## Measured results (2026-09-14, ShipOfTheseus)

| Stage | n_scored | accuracy | notes |
| --- | ---: | ---: | --- |
| G0 base | 16 | 0.0000 | `g0-eval/g0-base/metrics.json` |
| G1 paired base | 16 | 0.0000 | `g1-eval/base/metrics.json` |
| G1 paired SFT | 16 | 0.6250 | `g1-eval/sft/metrics.json` |

**Paired conclusion:** improved (10 improved, 0 regressed, 6 tied). See `comparison.json`, `report.md`, `training-results.json`.
