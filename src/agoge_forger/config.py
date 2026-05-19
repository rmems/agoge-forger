import yaml
from typing import List
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 0.0002
    num_train_epochs: int = 1
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

def load_config(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Flattened parsing to match yaml structure
    return ExperimentConfig(
        model_id=data['model_id'],
        trust_remote_code=data.get('trust_remote_code', True),
        dataset_path=data['dataset_path'],
        dataset_text_field=data.get('dataset_text_field', 'text'),
        output_dir=data.get('output_dir', 'adapters'),
        run_name=data.get('run_name', 'run'),
        quantization=QuantizationConfig(
            load_in_4bit=data.get('load_in_4bit', True),
            bnb_4bit_quant_type=data.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=data.get('bnb_4bit_compute_dtype', 'bfloat16'),
            bnb_4bit_use_double_quant=data.get('bnb_4bit_use_double_quant', True)
        ),
        training=TrainingConfig(
            max_seq_length=data.get('max_seq_length', 2048),
            batch_size=data.get('batch_size', 1),
            gradient_accumulation_steps=data.get('gradient_accumulation_steps', 8),
            gradient_checkpointing=data.get('gradient_checkpointing', True),
            learning_rate=data.get('learning_rate', 0.0002),
            num_train_epochs=data.get('num_train_epochs', 1),
            bf16=data.get('bf16', True),
            seed=data.get('seed', 42)
        ),
        lora=LoraConfigModel(
            lora_r=data.get('lora_r', 16),
            lora_alpha=data.get('lora_alpha', 32),
            lora_dropout=data.get('lora_dropout', 0.05),
            target_modules=data.get('target_modules', ["q_proj", "v_proj"]),
            target_modules_mode=data.get('target_modules_mode', 'auto_common')
        ),
        runtime=RuntimeConfig(
            save_safetensors=data.get('save_safetensors', True),
            allow_unsafe_serialization=data.get('allow_unsafe_serialization', False),
            max_shard_size=data.get('max_shard_size', "4GB")
        )
    )
