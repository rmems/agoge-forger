import torch

from ..logging import logger
from ..train.preflight import BYTES_PER_GB


def check_torch_env():
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        total_gib = torch.cuda.get_device_properties(0).total_memory / BYTES_PER_GB
        logger.info(f"Total VRAM: {total_gib:.2f} GiB")
    else:
        logger.warning("CUDA is NOT available. PyTorch will use CPU.")
