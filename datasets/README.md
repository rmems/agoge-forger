# Datasets

Place your training and evaluation datasets here.
`samples/tiny_sft.jsonl` is provided for smoke testing and validation.

Expected JSONL formats:
1. `{"text": "Your complete prompt and response here"}`
2. `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
3. `{"instruction": "...", "input": "...", "output": "..."}`
