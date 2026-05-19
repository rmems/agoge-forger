import os
from peft import PeftModel
from ..models.load import load_base_model
from ..logging import logger

def merge_adapter(base_model_id: str, adapter_path: str, out_dir: str):
    logger.info(f"Merging {adapter_path} into {base_model_id}")
    model, tokenizer = load_base_model(base_model_id, trust_remote_code=True, quant_config=None, bf16=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()
    
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving merged model to {out_dir}")
    merged_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
