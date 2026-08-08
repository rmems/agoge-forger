# Chat Completions Provider

A provider-neutral inference client for targeting local vLLM and other OpenAI-compatible chat-completions endpoints.

## Configuration

```yaml
provider: chat_completions
base_url: http://localhost:8000/v1
model: "your-model-name"
timeout_s: 120
stream: true
max_tokens: 512
temperature: 0.7
api_key: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | str | `chat_completions` | Provider identifier |
| `base_url` | str | `http://localhost:8000/v1` | vLLM or compatible endpoint base URL |
| `model` | str | `""` | Model name to send in requests |
| `timeout_s` | float | `120.0` | HTTP request timeout in seconds |
| `stream` | bool | `true` | Enable server-sent event streaming |
| `max_tokens` | int | `512` | Maximum tokens in the response |
| `temperature` | float | `0.7` | Sampling temperature |
| `api_key` | str | `""` | Optional API key (not logged) |

## Usage

### Non-streaming

```python
from agoge_forger.providers import ChatCompletionsConfig, ChatCompletionsClient

config = ChatCompletionsConfig(
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3.1-8B-Instruct",
    stream=False,
)

client = ChatCompletionsClient(config, run_name="my_run")
result = client.chat(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain QLoRA in one sentence."},
    ]
)

print(result.response_text)
print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
print(f"Latency: {result.latency_ms:.1f} ms")
```

### Streaming

```python
config = ChatCompletionsConfig(
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3.1-8B-Instruct",
    stream=True,
)

client = ChatCompletionsClient(config, run_name="stream_run")
result = client.chat([{"role": "user", "content": "Hello!"}])

print(result.response_text)
print(f"Time to first token: {result.time_to_first_token_ms:.1f} ms")
```

### Simple prompt helper

```python
result = client.chat_simple("What is Agoge?", system="Be concise.")
```

## vLLM Compatibility Smoke

`agoge smoke-vllm` sends a minimal chat-completion request to a vLLM or
OpenAI-compatible endpoint and writes structured results under `runs/<run_name>/`:

```bash
# Non-streaming smoke against a local vLLM server
agoge smoke-vllm --model HuggingFaceTB/SmolLM2-135M-Instruct

# Streaming smoke
agoge smoke-vllm --model my-model --stream

# Remote endpoint
agoge smoke-vllm \
  --base-url http://gpu-node:8000/v1 \
  --model my-model \
  --api-key "$OPENAI_API_KEY"

# Load prompts from a YAML prompt set
agoge smoke-vllm \
  --model my-model \
  --prompt-set configs/prompts/smoke.yaml \
  --run-name my_smoke

# Validate wiring without a running server
agoge smoke-vllm --model my-model --dry-run
```

### Flags and environment

| Flag | Environment variable | Default | Description |
|------|----------------------|---------|-------------|
| `--base-url` | `AGOGE_SMOKE_BASE_URL` | `http://localhost:8000/v1` | Endpoint base URL |
| `--model` | `AGOGE_SMOKE_MODEL` | (required) | Model name or path passed in the request |
| `--api-key` | `OPENAI_API_KEY` or `VLLM_API_KEY` | `""` | Bearer token for endpoints that require auth |
| `--stream` | - | `false` | Use server-sent event streaming |
| `--prompt` | - | `What is the capital of France?` | Single prompt |
| `--system` | - | `You are a helpful assistant.` | System message |
| `--prompt-set` | - | - | YAML file with `system` and `prompts` list |
| `--run-name` | - | `vllm_smoke` | Output directory under `runs/` |
| `--dry-run` | - | `false` | Write synthetic results without calling the endpoint |
| `--config` | - | - | YAML file containing a `ChatCompletionsConfig` |

See `docs/vllm_model_compatibility.md` for which model artifacts can be served
(base HF id, merged safetensors directory, or adapter + base).

### Output artifacts

| File | Description |
|------|-------------|
| `runs/<run_name>/smoke_vllm_result.json` | Full per-prompt `InferenceResult` objects with smoke status |
| `runs/<run_name>/results.jsonl` | Per-prompt benchmark-event lines |
| `runs/<run_name>/summary.md` | Human-readable summary with latency, TTFT, and token counts |
| `runs/<run_name>/raw/<request_id>.json` | Raw server response (written by `ChatCompletionsClient`) |

The command exits with code `0` when all prompts succeed (or are run in
`--dry-run`) and `1` when any prompt reports an error. Errors such as
`Connection refused`, `Request timed out`, or an HTTP status code are
preserved in the result files.

## InferenceResult

Every request returns an `InferenceResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `provider` | str | Provider name (`chat_completions`) |
| `base_url` | str | Endpoint URL |
| `model` | str | Model used |
| `request_id` | str | Unique request identifier |
| `prompt_hash` | str | SHA-256 hash of the prompt (first 16 chars) |
| `response_text` | str | Generated text |
| `reasoning_text` | str | Chain-of-thought reasoning (if available) |
| `finish_reason` | str | Stop reason (`stop`, `length`, etc.) |
| `input_tokens` | int | Prompt token count |
| `output_tokens` | int | Completion token count |
| `total_tokens` | int | Total token count |
| `latency_ms` | float | End-to-end request latency |
| `time_to_first_token_ms` | float | Time to first token (streaming) |
| `raw_response_path` | str | Path to saved raw response JSON |
| `error` | str | Error message (empty on success) |

## Raw Responses

All raw responses are written to `runs/<run_name>/raw/<request_id>.json`.

For streaming requests, the file contains a JSON object with a `"chunks"` key holding the array of SSE chunks.

## Targeting Local vLLM

To switch between endpoints, change only `base_url` and `model`:

```python
# Local vLLM
config = ChatCompletionsConfig(base_url="http://localhost:8000/v1", model="my-model")

# Remote vLLM
config = ChatCompletionsConfig(base_url="http://gpu-node:8000/v1", model="my-model")

# OpenAI-compatible (e.g. LiteLLM proxy)
config = ChatCompletionsConfig(base_url="http://proxy:4000/v1", model="gpt-4", api_key="sk-...")
```
