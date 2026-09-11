import torch

from agoge_forger.eval.contract import DecodingContract
from agoge_forger.eval.generate import PreparedTask, generate_completions
from agoge_forger.eval.serializers import code_repair_prompt
from agoge_forger.split_contract import SerializerBinding


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        extra = torch.tensor([[9, 8]], device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        ids = [1, 2, 3]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if skip_special_tokens:
            return " ".join(str(value) for value in values)
        return "raw:" + " ".join(str(value) for value in values)


def test_code_repair_prompt_is_fingerprintable():
    binding = SerializerBinding(implementation=code_repair_prompt)
    assert binding.serializer_id == "code-repair-prompt-v1"
    assert len(binding.serializer_sha256) == 64
    prompt = binding({"text": "Fix:\nanswer", "completion_start_char": 5})
    assert prompt == "Fix:\n"


def test_generate_completions_decodes_only_new_tokens():
    decoding = DecodingContract(do_sample=False, seed=7, max_new_tokens=8, temperature=0, top_p=1)
    tasks = (PreparedTask("t", "prompt", "expected", "ready", prompt_token_count=3),)
    records = generate_completions(_TinyModel(), _TinyTokenizer(), tasks, decoding)
    assert records[0].status == "ok"
    assert records[0].completion == "9 8"
    assert records[0].completion_token_count == 2
    assert records[0].prompt_token_count == 3
