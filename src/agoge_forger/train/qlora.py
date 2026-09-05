from .trainer import run_training


def train_qlora(config):
    # Honor YAML/config `load_in_4bit`. The model default and the 5080 QLoRA
    # recipe stay 4-bit; configs such as minicpm5_canary.yaml can opt in explicitly.
    run_training(config)
