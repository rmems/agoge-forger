"""Deterministic verification verdicts for reproducibility bundles."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

VERIFICATION_REPORT_VERSION: Literal["agoge.bundle-verification.v1"] = (
    "agoge.bundle-verification.v1"
)
BundleVerdict = Literal["pass", "fail"]


class BundleVerifyFormat(str, Enum):
    """Supported `agoge verify-bundle --format` renderings."""

    json = "json"
    table = "table"


@dataclass(frozen=True)
class BundleFailure:
    code: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class BundleVerificationReport:
    verdict: BundleVerdict
    bundle_path: str
    bundle_schema_version: str | None
    failures: tuple[BundleFailure, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERIFICATION_REPORT_VERSION,
            "verdict": self.verdict,
            "bundle_path": self.bundle_path,
            "bundle_schema_version": self.bundle_schema_version,
            "failures": [failure.as_dict() for failure in self.failures],
        }


def format_verification_table(report: BundleVerificationReport) -> str:
    rows = [
        ("verdict", report.verdict),
        ("bundle", escape_controls(report.bundle_path)),
        ("bundle_schema_version", report.bundle_schema_version or "-"),
        ("failures", str(len(report.failures))),
    ]
    width = max(len(label) for label, _ in rows) + 1
    lines = [f"{label:<{width}} {value}" for label, value in rows]
    for failure in report.failures:
        location = f"{escape_controls(failure.path)}: " if failure.path else ""
        lines.append(f"  {failure.code}: {location}{escape_controls(failure.message)}")
    return "\n".join(lines)


def escape_controls(text: str) -> str:
    """Render control, format, and surrogate characters as escapes."""

    unsafe = {"Cc", "Cf", "Cs"}

    def escaped(character: str) -> str:
        codepoint = ord(character)
        prefix, width = ("u", 4) if codepoint <= 0xFFFF else ("U", 8)
        return f"\\{prefix}{codepoint:0{width}x}"

    return "".join(
        escaped(character) if unicodedata.category(character) in unsafe else character
        for character in text
    )


def sort_failures(
    failures: tuple[BundleFailure, ...] | list[BundleFailure],
) -> tuple[BundleFailure, ...]:
    return tuple(sorted(failures, key=lambda item: (item.code, item.path or "", item.message)))
