from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


def write_complete_adapter_model(
    output_dir: Path,
    *,
    repository: str = "example/base-model",
    revision: str = "abcdef0123456789abcdef0123456789abcdef01",
) -> None:
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    base.name_or_path = repository
    adapter = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            revision=revision,
        ),
    )
    adapter.save_pretrained(output_dir, save_embedding_layers=False)
