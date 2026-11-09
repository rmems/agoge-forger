import os
from pathlib import Path

from peft import PeftModel

from ..artifacts.producer_provenance import require_producer_provenance
from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from ..eval.contract import ArtifactProducerProvenance
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


def require_empty_output_directory(out_dir: str) -> None:
    """Refuse a merge destination that already holds files.

    ``write_artifact_index`` seals every file it finds under the new adapter's
    provenance, and ``save_pretrained`` only removes stale ``-NNNNN-of-NNNNN``
    shards. A leftover ``model.safetensors`` from an earlier export would then be
    sealed as this merge's output and win at load time. An absent or empty
    directory is accepted; nothing is created here.
    """
    candidate = Path(out_dir).expanduser()
    if not os.path.lexists(candidate):  # absent (a dangling symlink still counts)
        return
    if not candidate.is_dir():
        raise ValueError(f"Merge output path exists and is not a directory: {candidate}")
    if any(candidate.iterdir()):
        raise ValueError(
            f"Refusing to merge into a non-empty output directory: {candidate}. "
            "Its existing files would be sealed into artifact_index.json under this "
            "adapter's provenance; choose a new or empty --out-dir."
        )


def _is_local_model_directory(base_model_id: str) -> bool:
    # Transformers loads a local directory whenever os.path.isdir() holds for the
    # given string, and then ignores ``revision`` entirely.
    return os.path.isdir(base_model_id) or Path(base_model_id).expanduser().is_dir()


def _is_sealed_local_directory(pinned_repo: str, snapshot: Path) -> bool:
    # The adapter was trained from this same local directory (an absolute path, never
    # a Hub id that merely collides with a cwd directory), so the sealed provenance
    # already names it; loading it seals no other identity.
    sealed_local = Path(pinned_repo).expanduser()
    if not (sealed_local.is_absolute() and sealed_local.is_dir()):
        return False
    return sealed_local.resolve(strict=True) == snapshot


def _is_sealed_cache_snapshot(snapshot: Path, repo_dir: str, pinned_revision: str) -> bool:
    return (snapshot.name, snapshot.parent.name, snapshot.parent.parent.name) == (
        pinned_revision,
        "snapshots",
        repo_dir,
    )


def _resolve_local_base(base_model_id: str, pinned_repo: str, pinned_revision: str) -> str:
    snapshot = Path(base_model_id).expanduser().resolve(strict=True)
    if _is_sealed_local_directory(pinned_repo, snapshot):
        return str(snapshot)
    expected_repo_dir = "models--" + pinned_repo.replace("/", "--")
    if not _is_sealed_cache_snapshot(snapshot, expected_repo_dir, pinned_revision):
        raise ValueError(
            f"Local base model directory {snapshot} cannot be tied to the sealed "
            f"producer_provenance {pinned_repo}@{pinned_revision}. Pass the Hub repo "
            f"id {pinned_repo!r} (it is loaded at the sealed revision, from the local "
            "cache when offline) or a Hugging Face cache snapshot directory "
            f"{expected_repo_dir}/snapshots/{pinned_revision}."
        )
    return str(snapshot)


def resolve_pinned_base_model(
    base_model_id: str, provenance: ArtifactProducerProvenance, revision: str | None
) -> str:
    """Return the base to load, refusing one not tied to the sealed provenance.

    The merged ``artifact_index.json`` copies the adapter's ``producer_provenance``
    (``base_model_name_or_path`` at ``revision``), so the base actually loaded must
    be that identity. A Hub repo id must equal the sealed one and is loaded at the
    sealed revision. A local directory carries no revision, so it is accepted only
    when it is the sealed local base directory itself, or a Hugging Face cache
    snapshot ``models--<org>--<name>/snapshots/<revision>`` of the sealed repo and
    revision.
    """
    pinned_repo = provenance.base_model_name_or_path
    pinned_revision = provenance.revision
    if revision not in (None, pinned_revision):
        raise ValueError(
            f"Adapter base revision {revision!r} does not match the sealed "
            f"producer_provenance revision {pinned_revision!r}"
        )
    if _is_local_model_directory(base_model_id):
        return _resolve_local_base(base_model_id, pinned_repo, pinned_revision)
    if base_model_id != pinned_repo:
        raise ValueError(
            f"Base model {base_model_id!r} does not match the adapter's sealed "
            f"producer_provenance base {pinned_repo!r}; refusing to seal that identity "
            "onto weights merged into a different base"
        )
    return base_model_id


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
    producer_provenance=None,
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

    provenance = require_producer_provenance(producer_provenance, adapter_path)
    # Both checks run before any weight is loaded: the merged index seals the
    # adapter's provenance, so the base must be that pinned identity and the
    # destination must not hold files this merge did not produce.
    pinned_base = resolve_pinned_base_model(base_model_id, provenance, revision)
    require_empty_output_directory(out_dir)

    model, tokenizer = load_base_model(
        pinned_base,
        trust_remote_code=trust_remote_code,
        quant_config=None,
        revision=provenance.revision,
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    # Validate the output path *before* touching the filesystem so a
    # rejected traversal path cannot create directories on disk. The
    # resolver also creates the directory via mkdir(parents=True, exist_ok=True).
    safe_out_dir = resolve_output_directory(out_dir)
    # Re-check after the (long) load: nothing may have landed there meanwhile.
    require_empty_output_directory(str(safe_out_dir))
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

    index_path = write_artifact_index(
        str(safe_out_dir),
        producer_provenance=provenance,
    )
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
    producer_provenance=None,
):
    source_adapter = resolve_export_source(
        run_dir=run_dir,
        adapter_path=adapter_path,
        allow_unsafe=allow_unsafe,
    )
    if base_model_id is None:
        resolved_base_model = infer_base_model_from_adapter(source_adapter)
    else:
        # An override must still be the sealed base (or its cache snapshot);
        # merge_adapter refuses anything else and always loads the pinned revision.
        resolved_base_model = base_model_id
    revision = infer_base_revision_from_adapter(source_adapter)
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
        producer_provenance=producer_provenance,
    )
