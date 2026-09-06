from ..artifacts.producer_provenance import producer_provenance_from_config
from .trainer import run_training


def train_qlora(config):
    # Honor YAML/config `load_in_4bit`. The model default and the 5080 QLoRA
    # recipe stay 4-bit; configs such as minicpm5_canary.yaml can opt in explicitly.
    run_training(config, producer_provenance=producer_provenance_from_config(config))
