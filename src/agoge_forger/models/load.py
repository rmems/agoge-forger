from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def get_bnb_config(quant_config):
    if not quant_config.load_in_4bit:
        return None
    compute_dtype = getattr(torch, quant_config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
    )

def load_base_model(model_id: str, trust_remote_code: bool, quant_config=None, bf16: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto"
    }

    if quant_config and quant_config.load_in_4bit:
        kwargs["quantization_config"] = get_bnb_config(quant_config)
    elif bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer
