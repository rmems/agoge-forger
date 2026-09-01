"""Dataset-source validation and path resolution for experiment configs."""

from pathlib import Path
from typing import Literal

from .path_safety import resolve_existing_path


def require_exclusive_dataset_source(dataset_path: str | None, frozen_fields: bool) -> None:
    if dataset_path is not None and frozen_fields:
        raise ValueError("dataset_path and frozen split configuration are mutually exclusive")
    if dataset_path is None and not frozen_fields:
        raise ValueError("configure either dataset_path or split_manifest_path with split_name")


def require_complete_frozen_source(
    manifest_path: str | None, split_name: Literal["train"] | None
) -> None:
    if (manifest_path is None) != (split_name is None):
        raise ValueError("split_manifest_path and split_name must be configured together")


def resolve_optional_input(value: object, config_path: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("dataset input path must be a string")
    raw_path = Path(value).expanduser()
    if not raw_path.is_absolute():
        raw_path = (config_path.parent / raw_path).resolve()
    return str(resolve_existing_path(str(raw_path), must_be_file=True))
