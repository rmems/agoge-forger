import os
from peft import PeftModel
from ..models.load import load_base_model
from ..logging import logger
from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index

def merge_adapter(base_model_id: str, adapter_path: str, out_dir: str, 
                  save_safetensors: bool = True, allow_unsafe: bool = False, max_shard_size: str = "4GB"):
    logger.info(f"Merging {adapter_path} into {base_model_id}")
    model, tokenizer = load_base_model(base_model_id, trust_remote_code=True, quant_config=None, bf16=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()
    
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving merged model to {out_dir}")
    merged_model.save_pretrained(out_dir, safe_serialization=save_safetensors, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(out_dir)
    
    if save_safetensors and not allow_unsafe:
        assert_no_unsafe_weight_bins(out_dir)
        
    index_path = write_artifact_index(out_dir)
    logger.info(f"Artifact index written to {index_path}")
