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
Admitted source lines are preserved byte-for-byte, with a final LF added only
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
file and a report; an empty input fails. After writing the report, tokenizer hashes
are rechecked, then the source hash is checked immediately before atomic directory
publication. These checks detect observed changes; they do not lock the inputs or
prevent a concurrent write after a check. Keep source and tokenizer files stable
throughout the command. Existing outputs are never overwritten.

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
