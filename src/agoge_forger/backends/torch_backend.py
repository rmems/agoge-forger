import torch
from ..logging import logger

def check_torch_env():
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is NOT available. PyTorch will use CPU.")
