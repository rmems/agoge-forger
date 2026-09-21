from pathlib import Path

from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.mixture_contract import MixtureCompositionSpec, load_mixture_spec
from agoge_forger.split_contract import canonical_json_bytes
from tests.mixture_composer_fixtures import compose_spec, freeze_source, write_sidecars
from tests.test_path_safety import _allowlist_ambient_tmp


def _write_spec(path: Path, spec: MixtureCompositionSpec) -> None:
    path.write_bytes(canonical_json_bytes(spec.model_dump(mode="json")) + b"\n")


def _invoke_compose(spec: Path, output: Path):
    return CliRunner().invoke(
        app,
        ["compose-mixture", "--spec", str(spec), "--output-dir", str(output)],
    )


def test_compose_mixture_cli_writes_frozen_snapshot(tmp_path):
    snapshot = freeze_source(tmp_path, "synthetic")
    write_sidecars(snapshot)
    spec = compose_spec(
        (("synthetic", snapshot, 1),),
        budget=10,
        experiment_arm="B",
    )
    spec_path = tmp_path / "mixture-spec.json"
    output = tmp_path / "mixture"
    _write_spec(spec_path, spec)

    result = _invoke_compose(spec_path, output)

    assert result.exit_code == 0, result.output
    assert (output / "mixture_manifest.json").is_file()
    assert (output / "mixture.jsonl").is_file()
    assert (output / "mixture_report.md").is_file()
    loaded = load_mixture_spec(spec_path)
    assert loaded.policy.experiment_arm == "B"


def test_compose_mixture_cli_refuses_existing_destination(tmp_path, caplog):
    snapshot = freeze_source(tmp_path, "synthetic")
    write_sidecars(snapshot)
    spec = compose_spec((("synthetic", snapshot, 1),), budget=10)
    spec_path = tmp_path / "mixture-spec.json"
    output = tmp_path / "mixture"
    output.mkdir()
    (output / "keep-me.txt").write_text("untouched")
    _write_spec(spec_path, spec)

    with caplog.at_level("ERROR", logger="agoge"):
        result = _invoke_compose(spec_path, output)

    assert result.exit_code == 1, result.output
    assert (output / "keep-me.txt").read_text() == "untouched"
    assert not (output / "mixture_manifest.json").exists()
    assert caplog.messages


def test_compose_mixture_cli_allows_allowlisted_temp_prefix(tmp_path, monkeypatch):
    ambient, real_tmp = _allowlist_ambient_tmp(tmp_path, monkeypatch)
    snapshot = freeze_source(tmp_path, "synthetic")
    write_sidecars(snapshot)
    spec = compose_spec((("synthetic", snapshot, 1),), budget=10)
    spec_path = tmp_path / "mixture-spec.json"
    output = ambient / "nested" / "mixture"
    _write_spec(spec_path, spec)

    result = _invoke_compose(spec_path, output)

    published = real_tmp / "nested" / "mixture"
    assert result.exit_code == 0, result.output
    assert (published / "mixture_manifest.json").is_file()
