# vLLM Model Compatibility Matrix

This document maps `agoge-forger` artifact types to the `--model` value you pass
to `agoge serve-vllm` and `agoge smoke-vllm`.

## Supported serve modes

| Artifact type | Serve command | `smoke-vllm --model` | Compatibility | Notes |
|---|---|---|---|---|
| Base Hugging Face model ID | `agoge serve-vllm --model <hf-id>` | `<hf-id>` | Supported if vLLM supports the architecture | vLLM downloads weights at serve time. Requires network and HF access. |
| Merged safetensors directory (local) | `agoge serve-vllm --model /path/to/merged/run` | `/path/to/merged/run` | Supported | Produced by `agoge merge-adapter` or `agoge export-final-model`. The directory must contain `config.json` and safetensors shards. |
| PEFT adapter + base model | `agoge serve-vllm --model <base-id> --extra-arg "--lora-modules=<name>=<adapter_path>"` | `<base-id>` | Partial; vLLM LoRA support is model-family specific | Use `agoge inspect-lora-targets` to confirm target modules. Requires the base model to load successfully. |

## Incompatible / known-broken pairs

| Model / setup | Why it may fail | Workaround |
|---|---|---|
| `HuggingFaceM4/tiny-random-LlamaForCausalLM` (CI random weights) | No chat template; random weights produce garbled outputs | Use a real instruction-tuned model such as `HuggingFaceTB/SmolLM2-135M-Instruct` for smoke tests. |
| QLoRA 4-bit adapter served directly | vLLM loads the base model; quantized adapter weights must be merged first | Run `agoge merge-adapter` or `agoge export-final-model` to produce a full-precision safetensors directory before serving. |
| Custom architectures not in vLLM | vLLM does not have the model class | Serve the merged `safetensors` with a vLLM-compatible `--model` / `--tokenizer` combination, or use the base `transformers` path in `agoge smoke-eval` instead. |

## Quick workflow

```bash
# 1. Export a merged model (after training is fixed / artifacts are available)
agoge export-final-model --run-dir adapters/my_run --out-dir merged/my_run

# 2. Serve it
agoge serve-vllm --model merged/my_run

# 3. Smoke it
agoge smoke-vllm --model merged/my_run --run-name smoke_merged
```

## Notes

- `agoge smoke-vllm` does **not** start the server. Start `agoge serve-vllm`
  separately or use an existing OpenAI-compatible endpoint.
- `HF_TOKEN` is never baked into configs; pass it at runtime when the model or
  tokenizer is gated.
- See `docs/chat_completions_providers.md` for the chat-completions client API
  and the full list of `agoge smoke-vllm` flags.
