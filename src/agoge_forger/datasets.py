import json

import numpy as np

from datasets import Dataset  # type: ignore[attr-defined]

from .logging import logger
from .path_safety import resolve_existing_path


def normalize_row(row, tokenizer=None, index=0):
    if "text" in row:
        if not isinstance(row["text"], str):
            raise ValueError(f"Line {index}: 'text' field must be a string.")
        return row
    elif "messages" in row:
        if not isinstance(row["messages"], list):
            raise ValueError(f"Line {index}: 'messages' must be a list.")
        for m in row["messages"]:
            if "role" not in m or "content" not in m:
                raise ValueError(f"Line {index}: messages must contain 'role' and 'content'.")
            if m["role"] not in ["user", "assistant", "system", "tool"]:
                raise ValueError(f"Line {index}: invalid role '{m['role']}'.")

        if tokenizer and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}
        else:
            text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in row["messages"]])
            return {"text": text}
    elif "instruction" in row:
        if not isinstance(row["instruction"], str):
            raise ValueError(f"Line {index}: 'instruction' must be a string.")
        text = f"Instruction: {row['instruction']}\n"
        if row.get("input"):
            text += f"Input: {row['input']}\n"
        text += f"Output: {row.get('output', '')}"
        return {"text": text}
    else:
        raise ValueError(
            f"Line {index}: Unknown format. Must contain 'text', 'messages', or 'instruction'."
        )


def load_jsonl_dataset(path: str, tokenizer=None) -> Dataset:
    dataset_path = resolve_existing_path(path, must_be_file=True)

    def gen():
        with dataset_path.open("r") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {i}: Invalid JSON - {e}")

                yield normalize_row(row, tokenizer, index=i)

    return Dataset.from_generator(gen)


def iter_normalized_rows(path: str):
    """Yield ``(line_number, normalized_row)`` without needing a tokenizer.

    ``normalize_row`` picks its output keys from the row *format* and never from
    the tokenizer, so preflight can see exactly what ``load_jsonl_dataset``
    would produce without first paying for ``load_base_model``. Only the
    ``messages`` chat-template *text* depends on the tokenizer; the keys do not.
    """
    dataset_path = resolve_existing_path(path, must_be_file=True)

    with dataset_path.open("r") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: Invalid JSON - {e}")

            yield i, normalize_row(row, None, index=i)


def dataset_stats(path: str, model_id: str, trust_remote_code: bool = False):
    from .models.load import load_base_model

    logger.info("Loading tokenizer for dataset stats...")
    _, tokenizer = load_base_model(model_id, trust_remote_code, quant_config=None, bf16=False)

    dataset = load_jsonl_dataset(path, tokenizer)
    lengths: list[int] = []

    for row in dataset:
        tokens = tokenizer(row["text"])["input_ids"]
        lengths.append(len(tokens))

    arr = np.asarray(lengths, dtype=np.int64)
    logger.info(f"Dataset Rows: {len(arr)}")
    logger.info(f"Max Tokens: {arr.max()}")
    logger.info(f"Mean Tokens: {arr.mean():.2f}")
    logger.info(f"Min Tokens: {arr.min()}")
