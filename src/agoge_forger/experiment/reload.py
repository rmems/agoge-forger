"""Clean-process reload of a G1 adapter.

The parent process asks a new interpreter to load the adapter. A missing
adapter file fails before any model download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .._strict_json import decode_json_object
from ..eval.contract import ArtifactIndexReference
from ..eval.generate import load_arm_model
from ..split_schema import canonical_json_bytes, sha256_file
from .arms import sft_arm
from .schema import ReloadObservation, load_experiment_contract

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def assert_adapter_files(adapter_dir: Path) -> None:
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json is missing: {adapter_dir}")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter_model.safetensors is missing: {adapter_dir}")


def clean_reload(
    *,
    contract_path: Path,
    adapter_dir: Path,
    runner: CommandRunner = subprocess.run,
) -> ReloadObservation:
    if not adapter_dir.is_dir():
        return ReloadObservation(status="failed", failure_reason="adapter directory is missing")
    completed = runner(
        [
            sys.executable,
            "-m",
            "agoge_forger.experiment.reload",
            "--contract",
            str(contract_path),
            "--adapter",
            str(adapter_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return _observation_from_process(completed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean-reload a G1 PEFT adapter")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args(argv)
    adapter_dir = Path(args.adapter)
    try:
        assert_adapter_files(adapter_dir)
        _load_adapter(Path(args.contract), adapter_dir)
    except Exception as exc:  # noqa: BLE001 - the parent records every reload failure
        _emit({"status": "failed", "failure_reason": str(exc) or exc.__class__.__name__})
        return 1
    _emit({"status": "reloaded"})
    return 0


def _load_adapter(contract_path: Path, adapter_dir: Path) -> None:
    index_path = adapter_dir / "artifact_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"artifact_index.json is missing: {adapter_dir}")
    contract = load_experiment_contract(contract_path)
    artifact = ArtifactIndexReference(
        kind="peft_adapter",
        artifact_index_path=str(index_path),
        artifact_index_sha256=sha256_file(index_path),
    )
    model, _tokenizer = load_arm_model(
        sft_arm(contract, artifact),
        artifact_root=str(adapter_dir),
        trust_remote_code=False,
    )
    del model


def _observation_from_process(completed: subprocess.CompletedProcess[str]) -> ReloadObservation:
    payload = _stdout_object(completed.stdout)
    status = payload.get("status")
    if completed.returncode == 0 and status == "reloaded":
        return ReloadObservation(status="reloaded")
    reason = payload.get("failure_reason")
    if not isinstance(reason, str) or not reason:
        reason = completed.stderr.strip() or f"clean reload exited {completed.returncode}"
    return ReloadObservation(status="failed", failure_reason=reason)


def _stdout_object(stdout: str) -> Mapping[str, object]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return decode_json_object(
            text.encode("utf-8"),
            "clean-reload stdout",
            object_label="reload",
        )
    except ValueError:
        return {"failure_reason": text}


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(dict(payload)) + b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
