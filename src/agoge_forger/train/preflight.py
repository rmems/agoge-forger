import torch
import re
from ..logging import logger

def check_cuda_available(required=True):
    if not torch.cuda.is_available():
        if required:
            raise RuntimeError("CUDA is required but not available.")
        else:
            logger.warning("CUDA is not available.")
            return False
    return True

def get_gpu_report():
    if not torch.cuda.is_available():
        return {}
    
    report = {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "allocated_vram_gb": torch.cuda.memory_allocated(0) / 1e9,
        "bf16_supported": torch.cuda.is_bf16_supported()
    }
    return report

def estimate_training_risk(config, gpu_report):
    if not gpu_report:
        return
        
    vram = gpu_report.get("total_vram_gb", 0)
    
    if vram <= 16.5:
        if not config.quantization.load_in_4bit:
            logger.warning("RISK: Training on <= 16GB VRAM without load_in_4bit is highly likely to OOM.")
        
        if config.training.batch_size > 1:
            logger.warning("RISK: Batch size > 1 on 16GB VRAM may cause OOM. Consider gradient_accumulation_steps instead.")
            
        if config.training.max_seq_length > 2048:
            logger.warning("RISK: max_seq_length > 2048 on 16GB VRAM may cause OOM.")

def validate_lora_targets_exist(model, lora_config):
    mode = getattr(lora_config, "target_modules_mode", "auto_common")
    requested_targets = lora_config.target_modules
    
    if mode == "discover_required" and not requested_targets:
        raise ValueError("target_modules_mode is discover_required but no target_modules were provided.")
        
    model_modules = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        model_modules.add(leaf)
        
    valid_targets = []
    
    if mode == "auto_common":
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        valid_targets = [t for t in common if t in model_modules]
        if not valid_targets:
            logger.warning("No common projection targets found. Proceeding with caution.")
    else:
        for t in requested_targets:
            # simple check if leaf target exists, or if regex matches
            found = False
            for name, _ in model.named_modules():
                if t in name or re.search(t, name):
                    found = True
                    break
            if found:
                valid_targets.append(t)
            else:
                logger.warning(f"Target module '{t}' not found in model.")
                if mode == "explicit":
                    raise ValueError(f"Explicit target module '{t}' does not exist in the model.")
                    
    if not valid_targets:
        raise ValueError("No valid LoRA target modules found or configured.")
        
    logger.info(f"Validated target modules: {valid_targets}")
    return valid_targets
