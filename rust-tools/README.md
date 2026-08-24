# Agoge Rust Tools

Optional workspace for high-performance data processing, telemetry, and experimental framework integrations. Rust is **not** a Python dependency of `agoge-forger` — it is a standalone toolchain that reads and writes the same files Python does, never imported or linked from the Python package. If you never install Rust, every Python workflow still works.

## Crates

- `agoge-cli`: the `agoge-cli` binary and the only entry point — subcommands `validate` and `benchgen`.
- `agoge-jsonl`: JSONL **syntax** validation (`validate_file`).
- `agoge-benchgen`: deterministic, seeded benchmark workload generation.
- `agoge-gguf`: GGUF inspection placeholder — a stub, not wired to anything yet.

## Building and running

Run everything from `rust-tools/` — the workspace root:

```bash
cargo build                      # build the workspace
cargo test                       # run the workspace test suite
cargo run -q -p agoge-cli -- --help
```

`-q` silences Cargo's own `Compiling…` / `Running…` lines (they go to stderr; the command's own output goes to stdout). From the repo root, `make rust-check` is the shortcut — it is just `cd rust-tools && cargo check`.

## `validate` — JSON syntax, not dataset schema

```bash
cargo run -q -p agoge-cli -- validate ../datasets/samples/tiny_sft.jsonl
```

```text
Validating ../datasets/samples/tiny_sft.jsonl ...
Validation successful.
```

Those two plain-text lines are a contract, not decoration: `docs/contracts/polyglot_boundaries.md` pins them, so scripts may match on them. Nothing else lands on stdout — no JSON, no JSONL.

**What it checks:** every non-empty line parses as JSON. Blank lines are skipped. Line numbers in errors are 1-indexed and count the blank lines too, so they match what your editor shows:

```bash
printf '%s\n' '{"text": "ok"}' '' '{"text": "trailing comma",}' > target/broken.jsonl
cargo run -q -p agoge-cli -- validate target/broken.jsonl
```

```text
Validating target/broken.jsonl ...
Error: Invalid JSON on line 3

Caused by:
    trailing comma at line 1 column 27
```

**What it deliberately does not check:** the dataset row schema. A row with none of `text`, `messages`, or `instruction` is still well-formed JSON, so it passes:

```bash
printf '%s\n' '{"prmopt": "typo in the key name", "answer": 42}' > target/wrong-schema.jsonl
cargo run -q -p agoge-cli -- validate target/wrong-schema.jsonl
```

```text
Validating target/wrong-schema.jsonl ...
Validation successful.
```

That is not a gap to fix — it is the boundary. `docs/contracts/dataset_jsonl.md` assigns row-schema ownership to Python, and duplicating those rules in Rust would give you two implementations to keep in sync and one of them silently wrong. Python enforces the schema when it loads a dataset (`src/agoge_forger/datasets.py`: `normalize_row`, reached from training and from `agoge dataset-stats`), and it raises with the same 1-indexed line numbering. Use `validate` as a fast syntax pre-check; use Python to find out whether the rows are *trainable*.

## `benchgen` — deterministic workload generation

`benchgen` builds a seeded set of prompt requests for smoke and serving benchmarks, so the same benchmark can be re-run tomorrow, on another machine, against a different server, and still be comparing like with like. It generates prompts only — it does not call a model.

| Flag | Default | Meaning |
|---|---|---|
| `--run-name` | *(required)* | Output directory name under `--runs-root` |
| `--count` | `32` | Number of requests; must be `1..=1000000` |
| `--seed` | `42` | PRNG seed; identical seeds give byte-identical output |
| `--workload` | `inference` | Label recorded on every row, and mixed into the seed |
| `--max-tokens` | `128` | `max_tokens` recorded on every row |
| `--stream` | off | Record `stream: true` on every row |
| `--runs-root` | `runs` | Root holding per-run output directories |

```bash
cargo run -q -p agoge-cli -- benchgen --run-name bench-demo --runs-root ../runs
```

```text
Generating workload for run bench-demo ...
Wrote ../runs/bench-demo/workload.jsonl (32 requests, seed 42).
Wrote ../runs/bench-demo/workload_manifest.json.
```

Two files land in `<runs-root>/<run-name>/`. `workload.jsonl` is one compact JSON object per line, every line newline-terminated — and ordinary JSONL, so `validate` reads it back:

```json
{"request_id":1,"workload":"inference","seed":42,"prompt":"Write a short guide to learning-rate warmup schedules.","max_tokens":128,"stream":false,"schema_version":1}
```

`workload_manifest.json` records the inputs that produced it (indent 2, matching the repo-wide JSON convention):

```json
{
  "run_name": "bench-demo",
  "workload": "inference",
  "count": 32,
  "seed": 42,
  "max_tokens": 128,
  "stream": false,
  "schema_version": 1,
  "generator": "agoge-benchgen",
  "generator_version": "0.1.0"
}
```

Every flag is independent, so a short streaming serving workload is one line:

```bash
cargo run -q -p agoge-cli -- benchgen --run-name serving-seed7 --count 8 --seed 7 \
  --workload serving --max-tokens 256 --stream --runs-root ../runs
```

```text
Generating workload for run serving-seed7 ...
Wrote ../runs/serving-seed7/workload.jsonl (8 requests, seed 7).
Wrote ../runs/serving-seed7/workload_manifest.json.
```

### Determinism

The same `(seed, count, workload, max_tokens, stream)` spec produces a **byte-for-byte identical** `workload.jsonl` — on any machine, any OS, any rustc version. A benchmark you cannot replay is not a benchmark. `max_tokens` and `stream` are part of that key because both are copied verbatim into every row; the prompt *text* alone depends on just `(seed, count, workload)`. `run_name` is not part of it either way: it names the directory and appears in the manifest, never in a row.

```bash
cargo run -q -p agoge-cli -- benchgen --run-name replay-a --seed 7 --count 8 --runs-root ../runs
cargo run -q -p agoge-cli -- benchgen --run-name replay-b --seed 7 --count 8 --runs-root ../runs
sha256sum ../runs/replay-a/workload.jsonl ../runs/replay-b/workload.jsonl
```

```text
339ce238ea41fb349c0a002481afe360fcc5d40b71f87dd8de23f44789e868ab  ../runs/replay-a/workload.jsonl
339ce238ea41fb349c0a002481afe360fcc5d40b71f87dd8de23f44789e868ab  ../runs/replay-b/workload.jsonl
```

That guarantee is bought by refusing the usual conveniences. The PRNG is a hand-rolled SplitMix64 rather than the `rand` crate, because `rand` does not promise a stable stream across versions — a routine dependency bump would silently change every workload ever generated. The workload-label hash is a hand-rolled FNV-1a for the same reason `DefaultHasher` cannot be used: it is explicitly unstable across rustc releases. The prompt fragments are `const` tables compiled into the binary, not an external file. The rows carry no timestamp, hostname, or path, JSON field order is fixed by struct declaration order, and nothing iterates a `HashMap`.

### Input limits

`--run-name` must be a single plain directory component — separators, `.`, `..`, and absolute paths are all rejected. It is checked **before** being joined onto `--runs-root`, and on the pre-resolution string, because canonicalizing first would consume the `..` and leave nothing to catch. That mirrors `path_safety.py` on the Python side. `--count` is capped at 1,000,000: every request is materialized in memory, so an unbounded count turns a typo into an OOM instead of an error message. All three failures exit non-zero and write nothing — not even the run directory.

```bash
cargo run -q -p agoge-cli -- benchgen --run-name ../escape --runs-root ../runs
cargo run -q -p agoge-cli -- benchgen --run-name .. --runs-root ../runs
cargo run -q -p agoge-cli -- benchgen --run-name demo --count 2000000 --runs-root ../runs
```

```text
Generating workload for run ../escape ...
Error: run_name must not contain a path separator: "../escape"
Generating workload for run .. ...
Error: run_name must not contain '.', '..', or a root component: ".."
Generating workload for run demo ...
Error: count 2000000 exceeds the maximum of 1000000
```

## Polyglot boundary

Languages here talk through **files only** — no FFI, no cross-language runtime dependency. Rust owns workload generation; Python owns dataset validation, training, and evaluation. The JSONL file *is* the interface, which is why `workload.jsonl` has a `schema_version` and why new fields may be added but never removed or renamed without a major version bump.

See [`../docs/contracts/polyglot_boundaries.md`](../docs/contracts/polyglot_boundaries.md) for the ownership matrix and [`../docs/contracts/dataset_jsonl.md`](../docs/contracts/dataset_jsonl.md) for the dataset row schema.

## Future Plans (Commented out in Cargo.toml)
- `burn`, `candle`, `dfdx`: Rust ML frameworks.
- `perf-rs`: Telemetry.
- `axolotl-rs`, `aprender`, `rl`: Training abstractions.
