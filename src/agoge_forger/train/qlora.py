from .trainer import run_training


def train_qlora(config):
    # Honor YAML/config `load_in_4bit`. The model default and the 5080 QLoRA
    # recipe stay 4-bit; smoke_test.yaml sets false so the tiny random Llama
    # is not forced through bitsandbytes.
    run_training(config)
