# Offline completion admission (RM-2156)

Before freezing a new dataset, count the exact completion-only training encoding:

```bash
uv run agoge admit-completions \
  --source candidates.jsonl \
  --tokenizer-dir /path/to/saved-fast-tokenizer \
  --max-seq-length 8192 \
  --output-dir admission-le8192
```

The tokenizer must already be saved locally with `tokenizer.json` and its special
token configuration. The command loads the saved fast tokenizer class without model weights,
executes no remote code, and makes no Hub requests. Use the same saved tokenizer
and special-token configuration as training. Admission shares training’s
missing-PAD-to-EOS fallback and right-padding setup. Source rows must contain exact
`text` and a Unicode-code-point `completion_start_char`; all metadata is retained.
In the default frozen-text mode, admitted source lines are preserved byte-for-byte, with a final LF added only
when missing, so decimal metadata is not rounded by JSON reserialization.
This guarantee applies to admission output only: the existing `freeze-split`
writer reserializes JSON numbers. Encode precision-sensitive metadata as strings
before the pipeline if exact decimal precision must survive freezing.
Rows with invalid boundaries, ambiguous offsets, no effective content targets,
or precomputed loss controls fail the entire command. No truncation is allowed.

The inclusive budget counts the trainer's encoding, including added special tokens
and terminal EOS. Output contains `admitted.jsonl`, a source-line-indexed
`decisions.jsonl`, and `admission_report.json` with the complete token-length
histogram, admission/rejection totals, source and output hashes, and local
tokenizer JSON file hashes. An all-over-budget input produces an empty admitted
file and a report; an empty input fails. Source and tokenizer hash checks must
still match before publication. Existing outputs are never overwritten.

This is a pre-freeze preparation artifact, not a frozen split or experiment
contract. For option A in RM-2156, run on the complete candidate source, then pass
`admitted.jsonl` to the existing `freeze-split` command with a new dataset version,
source identity, and leakage guards. Do not overwrite or relabel the old freeze.
Histograms can also be generated separately for existing train and held-out files
without changing their membership; do not combine those admitted subsets and call
that a new leakage-audited split. Pre-register the new experiment before G0.

The local tests use only synthetic text and a tiny saved tokenizer. This slice
does not establish real R2EGym counts, a new measured contract, GPU fit, or G1
quality; those remain separate RM-2156 acceptance work.

## Compact repair pairs (RM-2154)

For a new preparation artifact, use explicit pairs and opt in to the versioned
`search-replace-json-v1` rendering contract:

```json
{"canonical_id":"fruit-1","lineage_id":"fruit-source-1","instruction":"Change the fruit to pear.","before":"apple\n","after":"pear\n"}
```

```bash
uv run agoge admit-completions \
  --source repair-pairs.jsonl \
  --completion-format search-replace-json-v1 \
  --tokenizer-dir /path/to/saved-fast-tokenizer \
  --max-seq-length 8192 \
  --output-dir compact-admission
```

The serializer emits one JSON object with exactly two string fields, `search` and
`replace`. This is a versioned SEARCH/REPLACE dialect, not an Aider delimiter block.
It selects the span from the first changed line through the last changed line,
including one unchanged line on either side when available, and expands context
until the search occurs exactly once (including overlapping occurrences).
Distant edits remain one span; repeated context can require the entire source.
Compactness and admission are measured outcomes, not guarantees for every pair.

The prompt includes the instruction, original text as a JSON string, and the
output-format instructions. `completion_start_char` is the Unicode-code-point
length of that whole prompt, immediately before the opening `{` of the edit.
Only the serialized edit and terminal EOS are supervised. The `after` text never
appears in the prompt. Original `instruction`, `before`, and `after` fields are
replaced by rendered `text`, `completion_start_char`, `source_text` (the exact
original), and `completion_format`; identity and other metadata are retained.
Metadata JSON value spans retain their original decimal precision even though
rendered fields are newly serialized. Rendered-field collisions, alternate
message representations, and precomputed loss controls are rejected.

`apply_search_replace(source_text, completion)` performs only a string splice.
It refuses empty/no-op edits, malformed/duplicate/extra JSON keys, missing context,
and ambiguous context. It preserves exact Unicode, CRLF, and final-newline state.
The serializer verifies that applying its edit reconstructs `after` byte-for-byte
when UTF-8 encoded; it neither executes code nor opens/applies files. Empty
originals are unsupported; deletion of the entire nonempty original is supported.

Rendering runs **before** the parent's exact token-budget admission. The report
records the format version and hashes the raw input pairs and rendered admitted
rows. The default `frozen-text-v1` mode continues to accept existing rendered rows.
Neither mode changes a prior frozen split. Compact targets need a newly versioned
artifact and contract; any comparison retaining previous split membership must
reconcile canonical identities explicitly, rather than silently re-splitting it.

This slice does not wire an apply step into the evaluation harness, change scorer
semantics, register an experiment, produce ≥50 held-out tasks, or run G0/G1.
Those remain RM-2154 acceptance work; the pure text apply helper is available for
that later integration. Do not score these JSON edits as full-module completions.
