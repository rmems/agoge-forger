# CodeScene rule scoping

`code-health-rules.json` narrows Code Health rules for two scopes only. Everything else in
the repository keeps CodeScene's defaults.

| Path | Rules at weight 0 |
| --- | --- |
| `src/agoge_forger/_cli_*.py` | `Excess Number of Function Arguments`, `Missing Arguments Abstractions` |
| `src/agoge_forger/eval/rescore.py` | `Overall Code Complexity` |

## Why: Typer command modules

The `_cli_*.py` modules hold the Typer command functions, and in Typer a command's **parameters are
its CLI flags** — one parameter per `--flag`, each carrying its own help text:

| Command | Parameters | What they are |
| --- | --- | --- |
| `freeze-split` | 14 | `--source`, `--seed`, `--salt`, the three split weights, … |
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

### What is deliberately *not* scoped

`Complex Method`, `Overall Code Complexity`, `Large Method`, `Brain Method` and every other
rule remain at their defaults for the CLI modules. A CLI command body that grows complicated is
a genuine problem, and the split exists to keep those bodies thin — resolve, call, print.

## Why: `eval/rescore.py`

`Overall Code Complexity` scores a module's *mean* cyclomatic complexity against a threshold
of 4, so it also fires when a simple function is removed and the average of what remains
rises. That is what happened in #212: deleting the unused one-statement `read_generations`
wrapper left this module at a mean of 4.43 across seven functions, while every remaining
function stayed exactly as complex as it was. Restoring dead code or splitting working
functions to hold an average down would make the module worse.

Only the module-level mean is scoped. `Complex Method`, `Large Method`, `Brain Method` and
every other rule stay at their defaults for this file, so a function that actually grows
complicated still fails review. Remove this entry if a refactor brings the mean back under
the threshold.

## Maintenance

Regenerate from the official template in the CodeScene UI rather than hand-editing, if the
rule set ever needs to grow. Rule names must match CodeScene's biomarker names exactly; a
misspelled name is silently ignored rather than reported as an error.
