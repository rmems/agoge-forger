import json
import os

import torch
from peft import PeftModel

from ..logging import logger
from ..models.load import load_base_model
from ..train.checkpoints import infer_base_revision_from_adapter


def run_smoke_eval(
    base_model_id: str,
    adapter_path: str,
    run_name: str = "smoke_eval_run",
    trust_remote_code: bool = False,
    revision: str | None = None,
):
    if revision is None:
        revision = infer_base_revision_from_adapter(adapter_path)
    logger.info(f"Loading base model {base_model_id} and adapter {adapter_path}")
    model, tokenizer = load_base_model(
        base_model_id,
        trust_remote_code=trust_remote_code,
        quant_config=None,
        bf16=True,
        revision=revision,
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    prompts = ["What is Agoge?", "Explain QLoRA.", "What does GGUF stand for?"]

    results = []
    model.eval()
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50)
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Prompt: {p}\nResponse: {res}\n")
            results.append({"prompt": p, "response": res})

    os.makedirs(os.path.join("runs", run_name), exist_ok=True)
    with open(os.path.join("runs", run_name, "smoke_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
