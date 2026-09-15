# Frozen split report

- Source: `rmems/synthetic-factory@dcc8cc0b7757064d78d0fa445f1d787f8ab99cfc`
- Dataset version: `code-repair-v1-rm760-20260914`
- Source file: `outputs/code-repair/rm760-20260914-export/agoge/code_repair_v1.jsonl` (`80e42faf3e0033e765861aa33470866ace935a06fdb892828dc0e818710a8abb`)
- Source coverage: 155/155 records
- Split algorithm: `sha256-atomic-bucket-v1`
- Seed/salt: `20260908` / `python-repair-v1`

## Partitions

| Split | Records | Source-level SHA-256 |
|---|---:|---|
| train | 129 | `dca83e3655bf9d324b49895ccf0f40dcd7c94e2fab7f7769d3e059e2d6ac714b` |
| validation | 10 | `f8b8503d592689046cc615378293a524f3ccea3949748bcc7441477fe316d4ae` |
| held_out | 16 | `b5acdc7b4ebefce84ad3895aadcbf7ea4b3fe6e307fe80011e256d300ca18cb8` |

## Leakage guarantees

- canonical JSON content hashes do not cross splits
- canonical IDs do not cross splits
- source coordinates do not cross splits
- lineage IDs do not cross splits
- declared group IDs do not cross splits

## Exclusions

- None. Every valid source record is materialized exactly once.

## Limitations

- Deterministic gates do not claim semantic near-duplicate detection.
- Tokenizer and rendering statistics are immutable derivative sidecars.
