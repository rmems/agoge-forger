import pytest
import torch

from agoge_forger.eval._artifact_schema import ArtifactIndexReference
from agoge_forger.eval.contract import DecodingContract, EvaluationArm
from agoge_forger.eval.generate import PreparedTask, generate_completions, load_arm_model
from agoge_forger.eval.serializers import code_repair_prompt
from agoge_forger.split_contract import SerializerBinding

_REVISION = "abcdef0123456789abcdef0123456789abcdef01"
_SHA256 = "a" * 64


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


def _evaluation_arm(*, role: str, artifact=None, model_repository: str = "example/model"):
    return EvaluationArm(
        role=role,
        model_repository=model_repository,
        model_revision=_REVISION,
        artifact=artifact,
        tokenizer_repository="example/tokenizer",
        tokenizer_revision=_REVISION,
        tokenizer_sha256=_SHA256,
        serializer_id="messages-v1",
        serializer_version="1",
        serializer_sha256=_SHA256,
        logical_task_set_sha256=_SHA256,
        context_window=4096,
        truncation_policy="mark_unsupported",
        decoding=DecodingContract(
            do_sample=False, seed=17, max_new_tokens=128, temperature=0, top_p=1
        ),
        scoring_version="exact-match-v1",
    )


def _merged_arm() -> EvaluationArm:
    return _evaluation_arm(
        role="causal_sft",
        artifact=ArtifactIndexReference(
            kind="merged_model",
            artifact_index_path="artifact_index.json",
            artifact_index_sha256=_SHA256,
        ),
    )


def _stub_base_loader(monkeypatch):
    loaded: list[str] = []

    def fake_load(model_id: str, **_kwargs):
        loaded.append(model_id)
        return object(), object()

    monkeypatch.setattr("agoge_forger.eval.generate.load_base_model", fake_load)
    return loaded


def test_merged_artifact_rejects_pickle_weights_before_transformers_load(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "pytorch_model.bin").write_bytes(b"unsafe")
    loaded = _stub_base_loader(monkeypatch)
    with pytest.raises(RuntimeError, match="Unsafe weight binaries"):
        load_arm_model(_merged_arm(), artifact_root=str(merged), trust_remote_code=False)
    assert loaded == []


def test_merged_artifact_loads_when_pickle_weights_are_absent(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"safe")
    loaded = _stub_base_loader(monkeypatch)
    load_arm_model(_merged_arm(), artifact_root=str(merged), trust_remote_code=False)
    assert loaded == [str(merged)]


def test_peft_adapter_rejects_pickle_weights_in_adapter_dir(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"unsafe")
    loaded = _stub_base_loader(monkeypatch)
    arm = _evaluation_arm(
        role="causal_sft",
        artifact=ArtifactIndexReference(
            kind="peft_adapter",
            artifact_index_path="artifact_index.json",
            artifact_index_sha256=_SHA256,
        ),
    )
    with pytest.raises(RuntimeError, match="Unsafe weight binaries"):
        load_arm_model(arm, artifact_root=str(adapter), trust_remote_code=False)
    assert loaded == []
