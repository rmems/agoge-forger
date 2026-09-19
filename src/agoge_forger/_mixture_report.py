"""Rendering helpers for immutable mixture metadata."""

from .mixture_result_schema import MixtureManifest
from .split_schema import canonical_json_bytes


def manifest_bytes(manifest: MixtureManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"


def render_report(manifest: MixtureManifest) -> str:
    lines = _report_header(manifest)
    for source in manifest.sources:
        fill = "underfilled" if source.underfilled else "filled"
        lines.append(
            f"| `{source.source_id}` | {source.example_count} | {source.quota_tokens} | "
            f"{source.accepted_tokens} | {source.shortfall_tokens} | {fill} |"
        )
    lines.extend(_report_footer(manifest))
    return "\n".join(lines)


def _report_header(manifest: MixtureManifest) -> list[str]:
    tokenizer = manifest.tokenizer
    return [
        "# Frozen mixture report",
        "",
        f"- Experiment arm: `{manifest.policy.experiment_arm}`",
        f"- Consumed split: `{manifest.policy.consumed_split}`",
        f"- Algorithm: `{manifest.policy.algorithm_version}`",
        f"- Seed/salt: `{manifest.policy.seed}` / `{manifest.policy.salt}`",
        f"- Accepted-token budget: {manifest.policy.accepted_token_budget}",
        f"- Mixture SHA-256: `{manifest.artifact.sha256}`",
        f"- Tokenizer: `{tokenizer.tokenizer_id}@{tokenizer.tokenizer_revision}`",
        "",
        "## Sources",
        "",
        "| Source | Examples | Quota | Accepted tokens | Shortfall | Fill |",
        "|---|---:|---:|---:|---:|---|",
    ]


def _report_footer(manifest: MixtureManifest) -> list[str]:
    exclusion_lines = [
        f"- `{item.source_id}` `{item.canonical_id}`: {item.reason}" for item in manifest.exclusions
    ] or ["- None. Every available accepted-token group fit the allocated quotas."]
    return [
        "",
        "## Lineage guarantees",
        "",
        "- Canonical IDs do not cross sources or experiment arms",
        "- Lineage IDs do not cross train/validation/held-out or experiment arms",
        "- Declared group IDs do not cross sources or experiment arms",
        "- Exact content hashes do not cross sources or experiment arms",
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
