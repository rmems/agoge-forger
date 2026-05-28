import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ..datasets import load_jsonl_dataset
from ..models.load import load_base_model
from ..manifests import write_run_manifest
from ..logging import logger
from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from .preflight import check_cuda_available, get_gpu_report, estimate_training_risk, validate_lora_targets_exist

def run_training(config):
    check_cuda_available(required=True)
    gpu_report = get_gpu_report()
    logger.info(f"GPU Report: {gpu_report}")
    
    estimate_training_risk(config, gpu_report)
    
    logger.info(f"Loading {config.model_id} for run {config.run_name}")
    model, tokenizer = load_base_model(
        config.model_id, 
        config.trust_remote_code, 
        config.quantization, 
        config.training.bf16
    )
    
    if config.training.gradient_checkpointing:
        model.config.use_cache = False
    
    if config.quantization.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )
    
    target_modules = validate_lora_targets_exist(model, config.lora)
    
    peft_config = LoraConfig(
        r=config.lora.lora_r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    dataset = load_jsonl_dataset(config.dataset_path, tokenizer)
    logger.info(f"Dataset size: {len(dataset)}")
    
    out_dir = os.path.join(config.output_dir, config.run_name)
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        bf16=config.training.bf16,
        logging_steps=1,
        save_strategy="no",
        seed=config.training.seed,
        gradient_checkpointing=config.training.gradient_checkpointing
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field=config.dataset_text_field,
        max_seq_length=config.training.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving adapter to {out_dir}")
    trainer.model.save_pretrained(out_dir, safe_serialization=config.runtime.save_safetensors)
    tokenizer.save_pretrained(out_dir)
    
    if not config.runtime.allow_unsafe_serialization:
        assert_no_unsafe_weight_bins(out_dir)
        
    index_path = write_artifact_index(out_dir)
    logger.info(f"Artifact index written to {index_path}")
    
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"Max VRAM used: {vram_used:.2f} GB")
    
    metrics = {
        "max_vram_gb": vram_used,
        "gpu_report": gpu_report,
        "artifact_index": index_path
    }
    
    write_run_manifest(os.path.join("runs", config.run_name), config.model_dump(), metrics, model, tokenizer, dataset)
