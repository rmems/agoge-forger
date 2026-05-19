import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ..datasets import load_jsonl_dataset
from ..models.load import load_base_model
from ..manifests import write_run_manifest
from ..logging import logger

def run_training(config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training in agoge-forger.")
        
    logger.info(f"Loading {config.model_id} for run {config.run_name}")
    model, tokenizer = load_base_model(
        config.model_id, 
        config.trust_remote_code, 
        config.quantization, 
        config.training.bf16
    )
    
    if config.quantization.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )
    
    peft_config = LoraConfig(
        r=config.lora.lora_r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
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
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"Max VRAM used: {vram_used:.2f} GB")
    
    write_run_manifest(os.path.join("runs", config.run_name), config.dict(), {"max_vram_gb": vram_used})
