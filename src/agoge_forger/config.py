import yaml
from typing import Optional
from pydantic import BaseModel, Field

from .path_safety import resolve_existing_path


class TrainingConfig(BaseModel):
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 0.0002
    num_train_epochs: int = 1
    bf16: bool = True
    seed: int = 42
    save_steps: int = 50
    save_total_limit: Optional[int] = 2
    resume_from_latest_checkpoint: bool = False
    resume_checkpoint_path: Optional[str] = None


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class LoraConfigModel(BaseModel):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    target_modules_mode: str = "auto_common"


class RuntimeConfig(BaseModel):
    save_safetensors: bool = True
    allow_unsafe_serialization: bool = False
    max_shard_size: str = "4GB"
    disk_free_warning_gb: float = 20.0
    checkpoint_disk_buffer_gb: float = 8.0


class ExperimentConfig(BaseModel):
    model_id: str
    trust_remote_code: bool = False
    dataset_path: str
    dataset_text_field: str = "text"
    output_dir: str = "adapters"
    run_name: str = "run"

    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_config(yaml_path: str) -> ExperimentConfig:
    config_path = resolve_existing_path(yaml_path, must_be_file=True)
    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    # Validate loaded YAML structure
    if data is None or not isinstance(data, dict):
        raise ValueError(
            f"Invalid config file '{yaml_path}': expected a YAML mapping, got {type(data).__name__}"
        )
    if "dataset_path" not in data:
        raise ValueError(f"Invalid config file '{yaml_path}': missing required key 'dataset_path'")
    if "model_id" not in data:
        raise ValueError(f"Invalid config file '{yaml_path}': missing required key 'model_id'")

    # Resolve relative `dataset_path` entries against the directory of
    # the config file itself, not the current working directory, so
    # configs are portable and reproducible across environments.
    from pathlib import Path

    raw_dataset_path = Path(str(data["dataset_path"])).expanduser()
    if not raw_dataset_path.is_absolute():
        raw_dataset_path = (config_path.parent / raw_dataset_path).resolve()
    dataset_path = str(resolve_existing_path(str(raw_dataset_path), must_be_file=True))

    # Flattened parsing to match yaml structure
    return ExperimentConfig(
        model_id=data["model_id"],
        trust_remote_code=data.get("trust_remote_code", False),
        dataset_path=dataset_path,
        dataset_text_field=data.get("dataset_text_field", "text"),
        output_dir=data.get("output_dir", "adapters"),
        run_name=data.get("run_name", "run"),
        quantization=QuantizationConfig(
            load_in_4bit=data.get("load_in_4bit", True),
            bnb_4bit_quant_type=data.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=data.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_use_double_quant=data.get("bnb_4bit_use_double_quant", True),
        ),
        training=TrainingConfig(
            max_seq_length=data.get("max_seq_length", 2048),
            batch_size=data.get("batch_size", 1),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 8),
            gradient_checkpointing=data.get("gradient_checkpointing", True),
            learning_rate=data.get("learning_rate", 0.0002),
            num_train_epochs=data.get("num_train_epochs", 1),
            bf16=data.get("bf16", True),
            seed=data.get("seed", 42),
            save_steps=data.get("save_steps", 50),
            save_total_limit=data.get("save_total_limit", 2),
            resume_from_latest_checkpoint=data.get("resume_from_latest_checkpoint", False),
            resume_checkpoint_path=data.get("resume_checkpoint_path"),
        ),
        lora=LoraConfigModel(
            lora_r=data.get("lora_r", 16),
            lora_alpha=data.get("lora_alpha", 32),
            lora_dropout=data.get("lora_dropout", 0.05),
            target_modules=data.get("target_modules", ["q_proj", "v_proj"]),
            target_modules_mode=data.get("target_modules_mode", "auto_common"),
        ),
        runtime=RuntimeConfig(
            save_safetensors=data.get("save_safetensors", True),
            allow_unsafe_serialization=data.get("allow_unsafe_serialization", False),
            max_shard_size=data.get("max_shard_size", "4GB"),
            disk_free_warning_gb=data.get("disk_free_warning_gb", 20.0),
            checkpoint_disk_buffer_gb=data.get("checkpoint_disk_buffer_gb", 8.0),
        ),
    )
