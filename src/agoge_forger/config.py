import yaml
from typing import Any, List
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 0.0002
    num_train_epochs: int = 1
    warmup_steps: int = 0
    max_steps: int = -1
    save_steps: int = 500
    eval_steps: float = 0.0
    weight_decay: float = 0.0
    packing: bool = False
    train_on_completions: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    bf16: bool = True
    seed: int = 42

class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

class LoraConfigModel(BaseModel):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj"]
    target_modules_mode: str = "auto_common"

class RuntimeConfig(BaseModel):
    save_safetensors: bool = True
    allow_unsafe_serialization: bool = False
    max_shard_size: str = "4GB"

class ExperimentConfig(BaseModel):
    model_id: str
    trust_remote_code: bool = True
    dataset_path: str
    dataset_text_field: str = "text"
    output_dir: str = "adapters"
    run_name: str = "run"
    
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    return section if isinstance(section, dict) else {}

def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        # Unsloth sometimes uses strings like "unsloth" for checkpointing mode.
        return default
    return bool(value)

def load_config(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    training = _section(data, "training")
    lora = _section(data, "lora")
    quantization = _section(data, "quantization")
    runtime = _section(data, "runtime")

    model_id = data.get("model_id")
    dataset_path = data.get("dataset_path")
    if not model_id:
        raise ValueError(f"Missing required 'model_id' in {yaml_path}")
    if not dataset_path:
        raise ValueError(f"Missing required 'dataset_path' in {yaml_path}")

    return ExperimentConfig(
        model_id=model_id,
        trust_remote_code=data.get('trust_remote_code', True),
        dataset_path=dataset_path,
        dataset_text_field=data.get('dataset_text_field', 'text'),
        output_dir=data.get('output_dir', 'adapters'),
        run_name=data.get('run_name', 'run'),
        quantization=QuantizationConfig(
            load_in_4bit=quantization.get('load_in_4bit', data.get('load_in_4bit', True)),
            bnb_4bit_quant_type=quantization.get('bnb_4bit_quant_type', data.get('bnb_4bit_quant_type', 'nf4')),
            bnb_4bit_compute_dtype=quantization.get('bnb_4bit_compute_dtype', data.get('bnb_4bit_compute_dtype', 'bfloat16')),
            bnb_4bit_use_double_quant=quantization.get('bnb_4bit_use_double_quant', data.get('bnb_4bit_use_double_quant', True))
        ),
        training=TrainingConfig(
            max_seq_length=training.get('max_seq_length', data.get('max_seq_length', 2048)),
            batch_size=training.get('batch_size', data.get('batch_size', 1)),
            gradient_accumulation_steps=training.get('gradient_accumulation_steps', data.get('gradient_accumulation_steps', 8)),
            gradient_checkpointing=_coerce_bool(training.get('gradient_checkpointing', data.get('gradient_checkpointing', True)), True),
            learning_rate=training.get('learning_rate', data.get('learning_rate', 0.0002)),
            num_train_epochs=training.get('num_train_epochs', training.get('num_epochs', data.get('num_train_epochs', data.get('num_epochs', 1)))),
            warmup_steps=training.get('warmup_steps', data.get('warmup_steps', 0)),
            max_steps=training.get('max_steps', data.get('max_steps', -1)),
            save_steps=training.get('save_steps', data.get('save_steps', 500)),
            eval_steps=training.get('eval_steps', data.get('eval_steps', 0.0)),
            weight_decay=training.get('weight_decay', data.get('weight_decay', 0.0)),
            packing=_coerce_bool(training.get('packing', data.get('packing', False)), False),
            train_on_completions=_coerce_bool(training.get('train_on_completions', data.get('train_on_completions', False)), False),
            optim=training.get('optim', data.get('optim', 'adamw_torch')),
            lr_scheduler_type=training.get('lr_scheduler_type', data.get('lr_scheduler_type', 'linear')),
            bf16=_coerce_bool(training.get('bf16', data.get('bf16', True)), True),
            seed=training.get('seed', training.get('random_seed', data.get('seed', data.get('random_seed', 42))))
        ),
        lora=LoraConfigModel(
            lora_r=lora.get('lora_r', data.get('lora_r', 16)),
            lora_alpha=lora.get('lora_alpha', data.get('lora_alpha', 32)),
            lora_dropout=lora.get('lora_dropout', data.get('lora_dropout', 0.05)),
            target_modules=lora.get('target_modules', data.get('target_modules', ["q_proj", "v_proj"])),
            target_modules_mode=lora.get('target_modules_mode', data.get('target_modules_mode', 'auto_common'))
        ),
        runtime=RuntimeConfig(
            save_safetensors=runtime.get('save_safetensors', data.get('save_safetensors', True)),
            allow_unsafe_serialization=runtime.get('allow_unsafe_serialization', data.get('allow_unsafe_serialization', False)),
            max_shard_size=runtime.get('max_shard_size', data.get('max_shard_size', "4GB"))
        )
    )
