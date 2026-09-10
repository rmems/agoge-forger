# Frozen completion-only training

Set the flat YAML key `completion_only_loss: true` and `dataset_text_field: text`.
The default remains false, preserving full-sequence training. The example
`configs/minicpm5_code_repair.yaml.example` pins the MiniCPM5 base revision and
sets the pilot limit to 2,048 tokens. It is a template with a placeholder path:
copy it to a YAML config and set the path to an existing frozen export's
`splits/train.jsonl`. Runnable configs must reference real datasets; the loader
continues to refuse missing paths. This example does not authorize or launch training.

Every row must carry exact pre-rendered `text` and `completion_start_char`, a
strict integer Unicode code-point offset into that text. The completion must be
nonempty. Keep `canonical_id`, `lineage_id`, and optional `group_id` on frozen
rows; preprocessing preserves these and other metadata. For example:

```json
{"canonical_id":"repair-1","lineage_id":"source-1","text":"Fix:\nanswer","completion_start_char":5}
```

The boundary comes from rendering; no separator search or prompt reconstruction
occurs. A fast tokenizer must provide reliable character offsets. A token that
crosses the boundary is refused. Prompt and added BOS tokens receive no loss;
all completion content and the terminal EOS receive loss. A terminal EOS is
added only if absent. Padding is masked by TRL's installed language-modeling
collator. Rows with no content targets after causal shifting, unsupported added
special tokens, or more than `max_seq_length` tokens including EOS are refused
before trainer construction. There is no truncation option in this path.

Caller-provided `labels`, `assistant_masks`, and `seq_lengths` are reserved and
refused to prevent overriding verified supervision. Input token IDs and completion
masks are regenerated from the exact text and declared boundary.

The trainer receives pretokenized IDs, internally derived labels (`-100` for
masked tokens), and a backwards-compatible completion mask with explicit
`SFTConfig.completion_only_loss=True`. Explicit labels also support TRL versions
that build labels during dataset preparation instead of in the collator.
Before trainer construction,
`completion_preprocessing.json` records the boundary unit, no-truncation policy,
row and supervised-target counts, maximum token length, and SHA-256 over the
prepared rows (including metadata). The ordinary run manifest retains the
resolved training configuration. The evidence file proves preprocessing only;
it is not evidence that a training run completed.

`tests/test_trainer_trl_api.py` uses a local tiny model and tokenizer to inspect
the actual trainer dataloader and execute one bounded CPU step, without Hub
downloads or production training.
