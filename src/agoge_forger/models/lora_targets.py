import re
from .load import load_base_model
from ..logging import logger

def inspect_lora_targets(model_id: str, trust_remote_code: bool = True):
    logger.info(f"Finding potential LoRA targets in {model_id}...")
    model, _ = load_base_model(model_id, trust_remote_code, quant_config=None, bf16=True)
    
    targets = set()
    patterns = [r"q_proj", r"k_proj", r"v_proj", r"o_proj", r"gate_proj", r"up_proj", r"down_proj", r"linear", r"gka", r"kalman", r"ssm"]
    
    for name, module in model.named_modules():
        if any(re.search(p, name.lower()) for p in patterns):
            # Just grab the leaf name
            leaf = name.split(".")[-1]
            targets.add(leaf)
            
    logger.info("Candidate LoRA target modules matching typical patterns or custom GKA/SSM architectures:")
    for t in sorted(list(targets)):
        logger.info(f" - {t}")
    return targets
