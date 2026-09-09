# CodeScene rule scoping

`code-health-rules.json` narrows two Code Health rules for `src/agoge_forger/_cli_*.py`
only. Everything else in the repository keeps CodeScene's defaults.

## Why

Those modules hold the Typer command functions, and in Typer a command's **parameters are
its CLI flags** — one parameter per `--flag`, each carrying its own help text:

| Command | Parameters | What they are |
| --- | --- | --- |
| `freeze-split` | 14 | `--source`, `--seed`, `--salt`, the three split weights, … |
| `smoke-vllm` | 13 | `--model`, `--prompt`, `--stream/--no-stream`, … |
| `serve-vllm` | 9 | `--model`, `--host`, `--port`, `--dtype`, … |
| `export-final-model` | 8 | `--out-dir`, `--run-dir`, `--adapter-path`, … |
| `merge-adapter` | 5 | `--base-model`, `--adapter-path`, `--out-dir`, … |

`Excess Number of Function Arguments` and `Missing Arguments Abstractions` both measure a
real smell in ordinary code: a function taking so many arguments that its callers are hard
to read and its responsibilities have blurred. Neither applies here. These functions have
exactly one caller — Typer's argument parser — and collapsing their parameters into an
options object would not simplify anything; it would delete the CLI surface, because Typer
derives `--flag` names, types, defaults and `--help` text from the signature itself.

The rules stay fully enabled everywhere else, including the library code these commands
call into, where the smell they measure is real. `plan_cleanup` was reduced from six
parameters to two for exactly that reason.

## What is deliberately *not* scoped

`Complex Method`, `Overall Code Complexity`, `Large Method`, `Brain Method` and every other
rule remain at their defaults for these files. A CLI command body that grows complicated is
a genuine problem, and the split exists to keep those bodies thin — resolve, call, print.

## Maintenance

Regenerate from the official template in the CodeScene UI rather than hand-editing, if the
rule set ever needs to grow. Rule names must match CodeScene's biomarker names exactly; a
misspelled name is silently ignored rather than reported as an error.
