# Frozen completion-only training

Set the flat YAML key `completion_only_loss: true` and `dataset_text_field: text`.
The default remains false, preserving full-sequence training. The example
`configs/minicpm5_code_repair.yaml.example` pins the MiniCPM5 base revision and
sets the pilot limit to 2,048 tokens. Copy it to a YAML config and replace both
required placeholders: `dataset_path` with an existing frozen export's
`splits/train.jsonl`, and `target_modules` with verified LoRA targets for the
pinned model revision. Missing datasets and empty targets are refused;
`discover_required` does not automatically discover targets during training.

The existing discovery command loads model weights. When ready to inspect, use
a local snapshot of revision `156170697656c48f69915b33a2fb44110242187c`:

```sh
uv run agoge inspect-lora-targets --model-id /path/to/pinned/local/snapshot --out reports/lora-targets.json
```

This command has no `--revision` option. Inspect the reported candidates and
copy the verified module names into `target_modules`. Targets have not been
populated or model compatibility verified by this template; the example does
not authorize or launch training.

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
collator. Any completion content at token index zero is refused because it
lacks causal context, even if later content tokens could receive loss. A BOS
or actual prompt token can supply that context. Rows with no content targets
after causal shifting, unsupported added special tokens, or more than
`max_seq_length` tokens including EOS are refused
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
