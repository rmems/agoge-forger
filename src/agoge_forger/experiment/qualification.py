"""Load the model-compatibility evidence required before a measured Granite run."""

from __future__ import annotations

from pathlib import Path

from .._strict_json import decode_json_object
from ..split_schema import sha256_file
from .schema import QualificationReport


def load_qualification_report(path: Path) -> tuple[QualificationReport, str]:
    """Return the report and the SHA-256 of the exact bytes that were read."""

    resolved = path.expanduser().resolve(strict=True)
    document = decode_json_object(
        resolved.read_bytes(),
        str(resolved),
        object_label="qualification report",
    )
    report = QualificationReport.model_validate(document)
    return report, sha256_file(resolved)
