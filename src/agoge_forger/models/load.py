from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from ..logging import logger

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

def load_base_model(model_id: str, trust_remote_code: bool, quant_config=None, bf16: bool = True, 
                    revision: str = None, local_files_only: bool = False, attn_implementation: str = None,
                    torch_dtype_str: str = "auto", device_map: str = "auto"):
    
    logger.info(f"Loading tokenizer {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=trust_remote_code,
        revision=revision,
        local_files_only=local_files_only
    )
    
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        logger.info("Tokenizer has chat_template.")
        
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info("Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token and no eos_token. Cannot safely pad.")
            
    tokenizer.padding_side = "right"

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "revision": revision,
        "local_files_only": local_files_only
    }
    
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if quant_config and quant_config.load_in_4bit:
        kwargs["quantization_config"] = get_bnb_config(quant_config)
    else:
        if torch_dtype_str == "auto":
            kwargs["torch_dtype"] = "auto"
        elif torch_dtype_str == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype_str == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif torch_dtype_str == "float32":
            kwargs["torch_dtype"] = torch.float32

    logger.info(f"Loading model {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer
