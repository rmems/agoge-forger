from peft import PeftModel

from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from ..logging import logger
from ..models.load import load_base_model
from ..path_safety import resolve_output_directory
from ..train.checkpoints import (
    infer_base_model_from_adapter,
    infer_base_revision_from_adapter,
    is_adapter_artifact,
    resolve_export_source,
)


def merged_model_save_kwargs(*, max_shard_size: str = "4GB") -> dict[str, str]:
    """Kwargs for ``PreTrainedModel.save_pretrained`` after ``merge_and_unload``.

    Transformers 5 removed ``safe_serialization`` from the public signature (it is
    absorbed by ``**kwargs`` and ignored). Only pass keys that still bind cleanly.
    """
    return {"max_shard_size": max_shard_size}


def merge_adapter(
    base_model_id: str,
    adapter_path: str,
    out_dir: str,
    save_safetensors: bool = True,
    allow_unsafe: bool = False,
    max_shard_size: str = "4GB",
    trust_remote_code: bool = False,
    revision: str | None = None,
    infer_revision: bool = True,
):
    """Merge a LoRA adapter into the base model and write a shippable checkpoint.

    Under Transformers 5, merged ``PreTrainedModel`` weights are always written as
    safetensors (``safe_serialization`` was removed from ``save_pretrained``).
    ``save_safetensors=False`` is therefore rejected here rather than silently
    ignored. Pass ``allow_unsafe=True`` only to accept unsafe *input* adapters or
    skip the post-save ``.bin`` assert — not to request pickle weight files.
    """
    logger.info(f"Merging {adapter_path} into {base_model_id}")

    if not save_safetensors:
        raise ValueError(
            "save_safetensors=False is not supported for merged-model export under "
            "Transformers 5: PreTrainedModel.save_pretrained always writes safetensors. "
            "Leave save_safetensors=True. allow_unsafe=True only relaxes adapter-input "
            "and post-save .bin checks — it does not restore legacy .bin output."
        )

    # Library-callers safety net: reject `.bin` adapters (incl. mixed
    # `.safetensors`+`.bin`) before PeftModel.from_pretrained is invoked.
    # The CLI command also enforces this at the boundary; keeping the check
    # here protects direct library callers (e.g. `export_final_model`).
    if not allow_unsafe and not is_adapter_artifact(adapter_path, allow_unsafe=False):
        raise ValueError(
            f"Adapter at '{adapter_path}' is not a valid safetensors-only "
            f"adapter artifact. Mixed or pickle-based .bin weights are "
            f"rejected; pass allow_unsafe=True to override."
        )

    if revision is None and infer_revision:
        revision = infer_base_revision_from_adapter(adapter_path)

    model, tokenizer = load_base_model(
        base_model_id,
        trust_remote_code=trust_remote_code,
        quant_config=None,
        bf16=True,
        revision=revision,
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    # Validate the output path *before* touching the filesystem so a
    # rejected traversal path cannot create directories on disk. The
    # resolver also creates the directory via mkdir(parents=True, exist_ok=True).
    safe_out_dir = resolve_output_directory(out_dir)
    logger.info(f"Saving merged model to {safe_out_dir}")
    # Transformers 5: no safe_serialization kwarg on PreTrainedModel (always safetensors).
    # PeftModel.save_pretrained (adapter path in train/trainer.py) still accepts it.
    merged_model.save_pretrained(
        str(safe_out_dir),
        **merged_model_save_kwargs(max_shard_size=max_shard_size),
    )
    tokenizer.save_pretrained(str(safe_out_dir))

    if not allow_unsafe:
        assert_no_unsafe_weight_bins(str(safe_out_dir))

    index_path = write_artifact_index(str(safe_out_dir))
    logger.info(f"Artifact index written to {index_path}")


def export_final_model(
    out_dir: str,
    run_dir: str | None = None,
    adapter_path: str | None = None,
    base_model_id: str | None = None,
    save_safetensors: bool = True,
    allow_unsafe: bool = False,
    max_shard_size: str = "4GB",
    trust_remote_code: bool = False,
):
    source_adapter = resolve_export_source(
        run_dir=run_dir,
        adapter_path=adapter_path,
        allow_unsafe=allow_unsafe,
    )
    if base_model_id is None:
        resolved_base_model = infer_base_model_from_adapter(source_adapter)
        revision = infer_base_revision_from_adapter(source_adapter)
    else:
        # A caller-supplied base may not contain the adapter's commit SHA.
        resolved_base_model = base_model_id
        revision = None
    logger.info(f"Exporting final merged model from {source_adapter}")
    merge_adapter(
        resolved_base_model,
        source_adapter,
        out_dir,
        save_safetensors=save_safetensors,
        allow_unsafe=allow_unsafe,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
        revision=revision,
        infer_revision=False,
    )
