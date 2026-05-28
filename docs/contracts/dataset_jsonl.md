# Dataset JSONL Schema

Training datasets are stored as newline-delimited JSON (JSONL). Each line is a standalone JSON object representing one training example.

## File Location

```
datasets/samples/tiny_sft.jsonl   # smoke test
datasets/<name>.jsonl             # user datasets (referenced by config.dataset_path)
```

## Accepted Row Formats

### Format A — Plain Text

```json
{"text": "User: What is Agoge?\nAssistant: Agoge was the rigorous education..."}
```

| Field  | Type | Required | Description                      |
|--------|------|----------|----------------------------------|
| `text` | str  | Yes      | Complete prompt+response string  |

### Format B — Chat Messages

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

| Field      | Type  | Required | Description                       |
|------------|-------|----------|-----------------------------------|
| `messages` | list  | Yes      | Array of message objects          |

Each message object:

| Field      | Type | Required | Description                                  |
|------------|------|----------|----------------------------------------------|
| `role`     | str  | Yes      | One of: `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `content`  | str  | Yes      | Message text                                 |

If the tokenizer has a `chat_template`, it is applied. Otherwise, messages are joined as `"Role: content"`.

### Format C — Instruction

```json
{"instruction": "Define JAX.", "input": "", "output": "JAX is an autograd..."}
```

| Field         | Type | Required | Description                                  |
|---------------|------|----------|----------------------------------------------|
| `instruction` | str  | Yes      | The instruction or question                  |
| `input`       | str  | No       | Additional context (appended as `Input: <input>`) |
| `output`      | str  | No       | Expected response (appended as `Output: <output>`) |

## Normalization

All formats are normalized to `{"text": "..."}` before training:

1. **Plain text** → used as-is
2. **Chat messages** → tokenized via chat template, or joined as `"Role: content"` lines
3. **Instruction** → concatenated as `Instruction: <instruction>\nInput: <input>\nOutput: <output>`

## Validation Rules

- Rows missing all three keys (`text`, `messages`, `instruction`) raise `ValueError`
- `text` must be a string
- `messages` must be a list of objects with valid `role` values
- Error messages are 1-indexed by line number

## Owner

| Language | Responsibility                                              |
|----------|-------------------------------------------------------------|
| Python   | Reads, validates, normalizes, and trains on JSONL datasets  |
| Rust     | Reads and validates JSONL (via `agoge-jsonl`)               |
| Julia    | Reads pre-processed datasets (Parquet/CSV from Python path) |

## Compatibility

- Lines must be valid JSON objects
- UTF-8 encoding
- No trailing comma in JSON
- Empty lines are skipped
