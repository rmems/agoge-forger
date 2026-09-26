"""CLI exit codes for `agoge verify-bundle`."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from agoge_forger.cli import app
from tests.reproducibility_bundle_cases import write_valid_bundle


def _assert_clean_exit(result, code: int) -> None:
    assert result.exit_code == code, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_verify_bundle_cli_passes_valid_bundle(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    result = CliRunner().invoke(app, ["verify-bundle", str(bundle)])
    _assert_clean_exit(result, 0)
    report = json.loads(result.stdout)
    assert report["verdict"] == "pass"
    assert report["failures"] == []
    assert report["schema_version"] == "agoge.bundle-verification.v1"


def test_verify_bundle_cli_fails_mutated_bundle(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    (bundle / "stray.txt").write_text("nope\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["verify-bundle", str(bundle), "--format", "json"])
    _assert_clean_exit(result, 1)
    report = json.loads(result.stdout)
    assert report["verdict"] == "fail"
    assert any(item["code"] == "extra_file" for item in report["failures"])


def test_verify_bundle_cli_table_format(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    result = CliRunner().invoke(app, ["verify-bundle", str(bundle), "--format", "table"])
    _assert_clean_exit(result, 0)
    assert "verdict" in result.stdout
    assert "pass" in result.stdout


def test_verify_bundle_cli_missing_path_exits_one(tmp_path):
    missing = tmp_path / "absent"
    result = CliRunner().invoke(app, ["verify-bundle", str(missing)])
    _assert_clean_exit(result, 1)
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "fail"
    assert "error" in payload


def test_verify_bundle_is_listed_in_help():
    result = CliRunner().invoke(app, ["--help"])
    _assert_clean_exit(result, 0)
    assert "verify-bundle" in result.stdout
