#!/usr/bin/env python3
"""Normalize verified local Hermes traces into trajectory v1.1 JSONL.

Usage:
    python scripts/normalize_hermes_trajectories.py \\
      --input <hermes-jsonl> \\
      --run-manifest <manifest-json> \\
      --model-admission <admission-json> \\
      --output <canonical-jsonl> \\
      --report <decision-report-json>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.hermes_normalize import (  # noqa: E402
    NORMALIZER_VERSION,
    normalize_files,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("normalize_hermes_trajectories")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Hermes JSONL trace")
    parser.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="Hermes run manifest JSON",
    )
    parser.add_argument(
        "--model-admission",
        type=Path,
        required=True,
        help="local_model_admission_v1 report JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Canonical trajectory v1.1 JSONL",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="accepted/quarantined/rejected decision report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = normalize_files(
            input_path=args.input,
            run_manifest_path=args.run_manifest,
            model_admission_path=args.model_admission,
            output_path=args.output,
            report_path=args.report,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "normalized with %s: accepted=%s quarantined=%s rejected=%s → %s (report %s)",
        NORMALIZER_VERSION,
        report["accepted_count"],
        report["quarantined_count"],
        report["rejected_count"],
        args.output,
        args.report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
