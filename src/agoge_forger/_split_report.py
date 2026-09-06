"""Rendering helpers for immutable split metadata."""

from .split_schema import SPLIT_NAMES, SplitManifest, canonical_json_bytes


def manifest_bytes(manifest: SplitManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"


def render_report(manifest: SplitManifest) -> str:
    lines = _report_header(manifest)
    for split in SPLIT_NAMES:
        artifact = manifest.splits[split]
        lines.append(f"| {split} | {artifact.record_count} | `{artifact.sha256}` |")
    lines.extend(_report_footer(manifest))
    return "\n".join(lines)


def _report_header(manifest: SplitManifest) -> list[str]:
    return [
        "# Frozen split report",
        "",
        f"- Source: `{manifest.source.repository}@{manifest.source.revision}`",
        f"- Dataset version: `{manifest.source.dataset_version}`",
        f"- Source file: `{manifest.source.path}` (`{manifest.source.sha256}`)",
        f"- Source coverage: {manifest.source.record_count}/{manifest.source.record_count} records",
        f"- Split algorithm: `{manifest.split_policy.algorithm_version}`",
        f"- Seed/salt: `{manifest.split_policy.seed}` / `{manifest.split_policy.salt}`",
        "",
        "## Partitions",
        "",
        "| Split | Records | Source-level SHA-256 |",
        "|---|---:|---|",
    ]


def _report_footer(manifest: SplitManifest) -> list[str]:
    exclusion_lines = [f"- {item}" for item in manifest.exclusions] or [
        "- None. Every valid source record is materialized exactly once."
    ]
    return [
        "",
        "## Leakage guarantees",
        "",
        *[f"- {item}" for item in manifest.leakage_audit.deterministic_guarantees],
        "",
        "## Exclusions",
        "",
        *exclusion_lines,
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in manifest.limitations],
        "",
    ]
