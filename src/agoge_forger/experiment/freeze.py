"""Freeze the Granite 4.1 first-SFT contract before any measured baseline exists."""

from __future__ import annotations

from pathlib import Path

from .._atomic_file import publish_bytes_noreplace, write_fsynced_bytes
from .._token_provenance import SerializerBinding
from ..eval.contract import held_out_task_ids, logical_task_set_sha256
from ..eval.serializers import code_repair_prompt
from ..path_safety import resolve_absent_output_directory, resolve_existing_path
from ..split_schema import sha256_bytes, sha256_file
from ..split_validation import validate_split_manifest
from .qualification import load_qualification_report
from .schema import (
    INTERPRETATION_RULE,
    REPORT_PARENT_NAME,
    RESEARCH_QUESTION,
    BlockerAttestation,
    DecodingBudget,
    GraniteFirstSftContract,
    RegisteredBudget,
    SplitPin,
    contract_json_bytes,
    interpretation_rule_sha256,
)


def freeze_experiment_contract(
    *,
    experiment_id: str,
    output_dir: str | Path,
    split_manifest: str | Path,
    qualification_report: str | Path,
    agoge_commit: str,
) -> GraniteFirstSftContract:
    """Write an immutable contract. Refuses a second freeze of the same directory."""

    destination = resolve_absent_output_directory(str(output_dir))
    _require_report_destination(destination, experiment_id)
    manifest_path = resolve_existing_path(str(split_manifest), must_be_file=True)
    qualification_path = resolve_existing_path(str(qualification_report), must_be_file=True)
    qualification, qualification_sha256 = load_qualification_report(qualification_path)
    if qualification.verdict != "pass":
        raise ValueError(
            "measured Granite execution is blocked until model qualification verdict is pass"
        )
    manifest = validate_split_manifest(manifest_path)
    task_ids = held_out_task_ids(manifest)
    serializer = SerializerBinding(implementation=code_repair_prompt)
    contract = GraniteFirstSftContract(
        experiment_id=experiment_id,
        agoge_commit=agoge_commit,
        research_question=RESEARCH_QUESTION,
        model_revision=qualification.model_revision,
        tokenizer_repository=qualification.tokenizer_repository,
        tokenizer_revision=qualification.tokenizer_revision,
        tokenizer_sha256=qualification.tokenizer_sha256,
        serializer_sha256=serializer.serializer_sha256,
        split=SplitPin(
            source_repository=manifest.source.repository,
            source_revision=manifest.source.revision,
            dataset_version=manifest.source.dataset_version,
            split_manifest_path=str(manifest_path),
            split_manifest_sha256=sha256_file(manifest_path),
            train_sha256=manifest.splits["train"].sha256,
            validation_sha256=manifest.splits["validation"].sha256,
            held_out_sha256=manifest.splits["held_out"].sha256,
            logical_task_ids=task_ids,
            logical_task_set_sha256=logical_task_set_sha256(task_ids),
        ),
        budget=RegisteredBudget(target_modules=qualification.target_modules),
        decoding=DecodingBudget(),
        interpretation_rule=INTERPRETATION_RULE,
        interpretation_rule_sha256=interpretation_rule_sha256(),
        blockers=BlockerAttestation(
            qualification_report_path=str(qualification_path),
            qualification_report_sha256=qualification_sha256,
        ),
    )
    payload = contract_json_bytes(contract)
    publish_bytes_noreplace(
        destination / "experiment-contract.json",
        payload,
        refusal="refusing to overwrite experiment contract",
        writer=write_fsynced_bytes,
    )
    digest_line = f"{sha256_bytes(payload)}  experiment-contract.json\n".encode()
    publish_bytes_noreplace(
        destination / "experiment-contract.sha256",
        digest_line,
        refusal="refusing to overwrite experiment contract digest",
        writer=write_fsynced_bytes,
    )
    return contract


def _require_report_destination(destination: Path, experiment_id: str) -> None:
    if destination.name != experiment_id:
        raise ValueError(
            f"experiment directory name {destination.name!r} must equal "
            f"experiment id {experiment_id!r}"
        )
    if destination.parent.name != REPORT_PARENT_NAME:
        raise ValueError(
            f"experiment directory must live under {REPORT_PARENT_NAME}/, "
            f"got parent {destination.parent.name!r}"
        )
    if destination.exists():
        raise FileExistsError(f"experiment directory already exists: {destination}")
