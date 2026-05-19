from .load import load_base_model
from ..logging import logger

def inspect_model(model_id: str, trust_remote_code: bool = True):
    logger.info(f"Inspecting {model_id}...")
    model, _ = load_base_model(model_id, trust_remote_code, quant_config=None, bf16=True)
    
    logger.info(f"Architecture: {model.config.architectures}")
    logger.info(f"Parameters: {model.num_parameters() / 1e9:.2f} B")
    logger.info(f"Dtype: {model.dtype}")
    
    module_types = set([type(m).__name__ for _, m in model.named_modules()])
    logger.info(f"Module types found: {module_types}")
