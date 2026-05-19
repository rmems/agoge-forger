from transformers import AutoConfig
from ..logging import logger

try:
    from huggingface_hub import model_info
except ImportError:
    model_info = None

def get_model_config_metadata(model_id: str, trust_remote_code: bool = True, revision: str = None):
    logger.info(f"Fetching metadata for {model_id}")
    config = AutoConfig.from_pretrained(
        model_id, 
        trust_remote_code=trust_remote_code, 
        revision=revision
    )
    
    meta = {
        "model_type": getattr(config, "model_type", "unknown"),
        "architectures": getattr(config, "architectures", []),
        "torch_dtype": str(getattr(config, "torch_dtype", "unknown")),
        "vocab_size": getattr(config, "vocab_size", "unknown"),
        "hidden_size": getattr(config, "hidden_size", "unknown"),
        "num_hidden_layers": getattr(config, "num_hidden_layers", "unknown"),
        "num_attention_heads": getattr(config, "num_attention_heads", "unknown")
    }
    
    if model_info is not None:
        try:
            info = model_info(model_id, revision=revision)
            filenames = [sib.rfilename for sib in info.siblings]
            meta["has_safetensors"] = any(f.endswith(".safetensors") for f in filenames)
            meta["has_bin"] = any(f.endswith(".bin") for f in filenames)
            meta["has_custom_code"] = any(f.endswith(".py") for f in filenames)
        except Exception as e:
            logger.warning(f"Could not fetch hub metadata: {e}")
            
    return meta
