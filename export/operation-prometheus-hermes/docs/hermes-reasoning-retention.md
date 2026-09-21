# Hermes reasoning retention and independent verification

## Hidden reasoning is not a trainable view

Hermes traces may contain `<think>…</think>` blocks or equivalent hidden
reasoning fields (`reasoning`, `hidden_reasoning`, `thinking`, `thought`,
including case variants). Those spans are producer-internal. They must not
appear in canonical trajectory v1.1 events, software payloads, or other
trainable surfaces.

`scripts/normalize_hermes_trajectories.py` case-folds and strips hidden
reasoning before emit and records the retention policy on
`execution_provenance.reasoning_retention`. A record that still contains hidden
reasoning after stripping is rejected. Manifest values are sanitized and
scanned the same way.

Raw traces that retain hidden reasoning stay outside Git (see
[hermes-raw-trace-custody.md](hermes-raw-trace-custody.md)).

## Hermes completion is not task success

Hermes `completed` and `partial` flags are **execution metadata** only. They
must be JSON booleans and are copied onto `execution.producer_completed` /
`execution.producer_partial`. Missing, empty, or coerced values are rejected.

`terminal_disposition` is set exclusively from the frozen run-manifest
independent verifier (identity distinct from the Hermes producer, version,
outcome, and at least one validated artifact hash):

| Verifier outcome | `terminal_disposition` |
| --- | --- |
| pass / successful / verified | `successful` |
| fail / failed / error | `failed` |
| interrupted / timeout / incomplete | `interrupted` |
| missing / null / unknown | quarantined as unverifiable |

A run with `completed=true` and a failing verifier is a **failed** trajectory.
It cannot be emitted as `successful`. Trace-embedded verifier objects that
conflict with the manifest are rejected.

## Deterministic normalization

Repeated invocation on the same frozen `--input`, `--run-manifest`, and
`--model-admission` bytes must produce byte-identical `--output` JSONL and
`--report` JSON. Output objects are canonical sorted JSON written as UTF-8
bytes with stable separators and explicit LF. Secret query keys are dropped
from embedded URLs before emit; credential-bearing URL paths fail closed.
