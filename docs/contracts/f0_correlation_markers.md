# F0 correlation markers (`bkl.f0_correlation.v1`)

Agoge emits the Agoge-owned side of the
[`blackwell-kernel-lab` F0 correlation schema](https://github.com/rmems/blackwell-kernel-lab/blob/main/docs/f0-correlation-schema.md).
BKL owns GPU samples and Nsight/CUPTI collectors. Agoge does not infer GPU
load and BKL must not infer training phase from utilization.

## Artifacts

Written under `runs/<run_name>/telemetry/` (the `runs/` tree is gitignored):

| File | Kind | When |
| --- | --- | --- |
| `agoge-markers.jsonl` | `agoge_marker` | Runs with marker emission enabled (cheap JSONL append). Disable with `telemetry.emit_markers: false`. |
| `profile-window-request.json` | `agoge_profile_window_request` | Every run. Records whether a window was requested and which backends are unsupported. |
| `profile-window.json` | `agoge_profile_window` | Only when a window is enabled. Compact summary; no chrome traces in git. |

`manifest.json` `metrics.telemetry` points only at artifacts published by the
current session. Its `markers` and `profile_window` paths are `null` when the
corresponding artifact is not published. Marker write failures are logged and
never abort training.

## Envelope

Markers use `schema_version: bkl.f0_correlation.v1`, `agoge_run_id`, host,
GPU identity (nulls when unknown — never empty strings treated as a device),
RFC 3339 `timestamp_utc` ending in `Z`, and `monotonic_ns` for local order.

`model_revision` is the pinned Hub revision, or `unpinned` when the config
did not pin one. That string is an explicit gap, not a guessed SHA.

## Requesting a window

A window is `run_id` + `phase` + inclusive `start_step`–`end_step`.

```bash
AGOGE_RUN_ID=run_minicpm5_canary_001 \
AGOGE_PROFILE_WINDOW=train:1-1 \
uv run agoge train-qlora --config configs/minicpm5_canary.yaml
```

Equivalent CLI:

```bash
uv run agoge train-qlora \
  --config configs/minicpm5_canary.yaml \
  --run-id run_minicpm5_canary_001 \
  --profile-window train:1-1
```

Telemetry is resolved once with this precedence: CLI flags, then environment,
then YAML, then defaults. Environment values replace YAML values even when the
YAML value is non-default; CLI values replace both. Ordinary configs do not
enable the profiler.

YAML form (optional):

```yaml
telemetry:
  run_id: run_minicpm5_canary_001
  emit_markers: true
  profile_window:
    enabled: true
    phase: train
    start_step: 1
    end_step: 1
    backend: torch
```

`backend` defaults to `torch`. `nsight_systems`, `nsight_compute`, and
`cupti` are recorded as `unsupported` from Agoge (BKL may attach them around
the same `profile_window_id`). Unknown backends are also `unsupported`; Agoge
does not silently fall back.

## Join

Join GPU samples to markers with `agoge_run_id` + `host.hostname` + GPU
identity, taking the latest marker with `monotonic_ns <= sample.monotonic_ns`
(see BKL `tools/check_f0_correlation.py`). Samples inside a window should
carry `profile_window_ref` equal to Agoge's `profile_window_id`
(`{run_id}:{phase}:{start}-{end}`).

CPU fixtures: `tests/fixtures/f0-correlation/`.

## Window completion fields

`profile_window_id`, `start_step`, and `end_step` retain the requested inclusive
range. `actual_end_step` records the final captured optimizer step and is `null`
when the requested start step was never reached. `complete` is true only when
the requested end step was captured. The monotonic bounds are both required
before a GPU sample can join to a window.
