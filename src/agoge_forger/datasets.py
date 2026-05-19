import json
from datasets import Dataset
from .logging import logger

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
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
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
        raise ValueError(f"Line {index}: Unknown format. Must contain 'text', 'messages', or 'instruction'.")

def load_jsonl_dataset(path: str, tokenizer=None) -> Dataset:
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: Invalid JSON - {e}")
            
            normalized = normalize_row(row, tokenizer, index=i)
            data.append(normalized)
    
    return Dataset.from_list(data)

def dataset_stats(path: str, model_id: str):
    from .models.load import load_base_model
    import numpy as np
    
    logger.info("Loading tokenizer for dataset stats...")
    _, tokenizer = load_base_model(model_id, trust_remote_code=True, quant_config=None, bf16=False)
    
    dataset = load_jsonl_dataset(path, tokenizer)
    lengths = []
    
    for row in dataset:
        tokens = tokenizer(row["text"])["input_ids"]
        lengths.append(len(tokens))
        
    lengths = np.array(lengths)
    logger.info(f"Dataset Rows: {len(lengths)}")
    logger.info(f"Max Tokens: {lengths.max()}")
    logger.info(f"Mean Tokens: {lengths.mean():.2f}")
    logger.info(f"Min Tokens: {lengths.min()}")
