from ..logging import logger
from .trainer import run_training


def train_lora(config):
    logger.warning(
        "Running standard LoRA (no 4-bit quantization). Ensure your VRAM can handle this model size!"
    )
    config.quantization.load_in_4bit = False
    run_training(config)
