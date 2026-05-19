import json
from datasets import Dataset

def normalize_row(row, tokenizer=None):
    if "text" in row:
        return row
    elif "messages" in row:
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            return {"text": text}
        else:
            # Fallback formatting
            text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in row["messages"]])
            return {"text": text}
    elif "instruction" in row:
        text = f"Instruction: {row['instruction']}\n"
        if row.get("input"):
            text += f"Input: {row['input']}\n"
        text += f"Output: {row.get('output', '')}"
        return {"text": text}
    return row

def load_jsonl_dataset(path: str, tokenizer=None) -> Dataset:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            data.append(normalize_row(row, tokenizer))
    
    return Dataset.from_list(data)
