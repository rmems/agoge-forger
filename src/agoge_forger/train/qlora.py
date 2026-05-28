from .trainer import run_training

def train_qlora(config):
    # QLoRA explicitly requires 4bit
    config.quantization.load_in_4bit = True
    run_training(config)
