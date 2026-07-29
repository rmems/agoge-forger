from peft import PeftModel
from ..models.load import load_base_model
from ..logging import logger
from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from ..path_safety import resolve_output_directory
from ..train.checkpoints import (
    infer_base_model_from_adapter,
    is_adapter_artifact,
    resolve_export_source,
)


def merge_adapter(
    base_model_id: str,
    adapter_path: str,
    out_dir: str,
    save_safetensors: bool = True,
    allow_unsafe: bool = False,
    max_shard_size: str = "4GB",
    trust_remote_code: bool = False,
):
    logger.info(f"Merging {adapter_path} into {base_model_id}")

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

    model, tokenizer = load_base_model(
        base_model_id, trust_remote_code=trust_remote_code, quant_config=None, bf16=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    # Validate the output path *before* touching the filesystem so a
    # rejected traversal path cannot create directories on disk. The
    # resolver also creates the directory via mkdir(parents=True, exist_ok=True).
    safe_out_dir = resolve_output_directory(out_dir)
    logger.info(f"Saving merged model to {safe_out_dir}")
    merged_model.save_pretrained(
        str(safe_out_dir), safe_serialization=save_safetensors, max_shard_size=max_shard_size
    )
    tokenizer.save_pretrained(str(safe_out_dir))

    if save_safetensors and not allow_unsafe:
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
    resolved_base_model = base_model_id or infer_base_model_from_adapter(source_adapter)
    logger.info(f"Exporting final merged model from {source_adapter}")
    merge_adapter(
        resolved_base_model,
        source_adapter,
        out_dir,
        save_safetensors=save_safetensors,
        allow_unsafe=allow_unsafe,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
    )
