from pathlib import Path

from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.split_contract import canonical_json_bytes


def _write_source(path: Path, count: int = 90) -> None:
    rows = [
        {
            "canonical_id": f"sample-{index:03d}",
            "lineage_id": f"lineage-{index // 2:03d}",
            "group_id": f"family-{index // 3:03d}",
            "text": f"Explain deterministic sample {index} with unique evidence {index * 17}.",
        }
        for index in range(count)
    ]
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def test_freeze_split_cli_does_not_precreate_destination(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "nested" / "snapshot"
    _write_source(source)

    result = CliRunner().invoke(
        app,
        [
            "freeze-split",
            "--source",
            str(source),
            "--source-path",
            "data/curated.jsonl",
            "--output-dir",
            str(output),
            "--source-repository",
            "rmems/synthetic-factory",
            "--source-revision",
            "0123456789abcdef0123456789abcdef01234567",
            "--dataset-version",
            "curated-sft-v1",
            "--seed",
            "20260830",
            "--salt",
            "agoge-issue-99-v1",
            "--train-weight",
            "6",
            "--validation-weight",
            "2",
            "--held-out-weight",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "split_manifest.json").is_file()
    assert (output / "split_report.md").is_file()
