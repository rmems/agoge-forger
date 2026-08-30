#!/usr/bin/env python3
"""Materialize an immutable three-way SFT split from a pinned local source."""

from __future__ import annotations

import argparse

from agoge_forger.split_contract import materialize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Versioned curated source JSONL")
    parser.add_argument("--output-dir", required=True, help="New immutable snapshot directory")
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--train-weight", type=int, default=80)
    parser.add_argument("--validation-weight", type=int, default=10)
    parser.add_argument("--held-out-weight", type=int, default=10)
    parser.add_argument("--canonical-id-field", default="canonical_id")
    parser.add_argument("--lineage-id-field", default="lineage_id")
    parser.add_argument("--group-id-field", default="group_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_split(
        source_path=args.source,
        output_dir=args.output_dir,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
        dataset_version=args.dataset_version,
        seed=args.seed,
        salt=args.salt,
        train_weight=args.train_weight,
        validation_weight=args.validation_weight,
        held_out_weight=args.held_out_weight,
        canonical_id_field=args.canonical_id_field,
        lineage_id_field=args.lineage_id_field,
        group_id_field=args.group_id_field,
    )
    print(f"wrote {args.output_dir}/split_manifest.json")
    print(f"wrote {args.output_dir}/split_report.md")
    print(
        "counts: "
        + ", ".join(
            f"{name}={manifest.splits[name].record_count}"
            for name in ("train", "validation", "held_out")
        )
    )


if __name__ == "__main__":
    main()
