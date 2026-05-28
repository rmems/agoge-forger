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
result = client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain QLoRA in one sentence."},
])

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

For streaming requests, the file contains an array of SSE chunks.

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
