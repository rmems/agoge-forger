# Frozen split report

- Source: `R2E-Gym/R2EGym-SFT-Trajectories@a9b3345a097eb4961adc8a32db35db0e9105cb6272a6211c3839b19856e99cb2`
- Dataset version: `r2egym-sft-cap512-last-assistant-v1`
- Source file: `admitted/r2egym_sft_cap512_last_assistant_v1.jsonl` (`f8cdb2e0ff479931aa9d37882544a168a266e8a8f22e993825440cfbacc159a8`)
- Source coverage: 512/512 records
- Split algorithm: `sha256-atomic-bucket-v1`
- Seed/salt: `20260908` / `r2egym-sft-cap512-last-assistant-v1`

## Partitions

| Split | Records | Source-level SHA-256 |
|---|---:|---|
| train | 411 | `23b182b267c6a76b4bcbbab7d54ac52d509cc7fe20e6186f0bc3d2dac891d3a0` |
| validation | 46 | `63ea1cf1566b8978ab6234fab31a5b8fd2a22d97f8dae210409e4d65da3b46c3` |
| held_out | 55 | `0451f93ebc57b16bf0fd458b34a721b75b106f58d9b373f81600d1924c58a8bf` |

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
