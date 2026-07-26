"""Tests for the artifact path-safety helpers in ``scripts/smoke_test.py``.

``scripts/`` is not an importable package, so the module under test is
loaded directly from its file path via ``importlib``.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SMOKE_TEST_PATH = Path(__file__).resolve().parent.parent / "scripts" / "smoke_test.py"


def _load_smoke_test_module():
    spec = importlib.util.spec_from_file_location("scripts_smoke_test", _SMOKE_TEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke_test = _load_smoke_test_module()


class TestSafeArtifactPath:
    def test_valid_basename_joins_under_out_dir(self, tmp_path):
        result = smoke_test._safe_artifact_path(tmp_path, "manifest.json")
        assert result == tmp_path / "manifest.json"

    def test_accepts_all_declared_artifact_constants(self, tmp_path):
        names = [
            smoke_test._ARTIFACT_MANIFEST,
            smoke_test._ARTIFACT_PROVIDER,
            smoke_test._ARTIFACT_USAGE_BEFORE,
            smoke_test._ARTIFACT_USAGE_AFTER,
            smoke_test._ARTIFACT_USAGE_DELTA,
            smoke_test._ARTIFACT_RESULTS,
            smoke_test._ARTIFACT_SUMMARY,
        ]
        for name in names:
            result = smoke_test._safe_artifact_path(tmp_path, name)
            assert result.parent == tmp_path
            assert result.name == name

    def test_rejects_empty_name(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, "")

    def test_rejects_dot(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, ".")

    def test_rejects_dotdot(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, "..")

    def test_rejects_relative_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, "../../etc/passwd")

    def test_rejects_nested_subpath(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, "sub/name.json")

    def test_rejects_backslash(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, "sub\\name.json")

    def test_rejects_absolute_path(self, tmp_path):
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._safe_artifact_path(tmp_path, "/etc/passwd")


class TestJailedOutputDir:
    def test_relative_dir_under_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = smoke_test._jailed_output_dir("smoke_output")
        assert result == tmp_path.resolve() / "smoke_output"

    def test_nested_relative_dir_under_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = smoke_test._jailed_output_dir("a/b/c")
        assert result == tmp_path.resolve() / "a" / "b" / "c"

    def test_dot_returns_cwd_itself(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = smoke_test._jailed_output_dir(".")
        assert result == tmp_path.resolve()

    def test_absolute_path_inside_cwd_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        inside = tmp_path / "nested"
        result = smoke_test._jailed_output_dir(str(inside))
        assert result == tmp_path.resolve() / "nested"

    def test_rejects_empty_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="must not be empty"):
            smoke_test._jailed_output_dir("")

    def test_rejects_parent_traversal_and_does_not_create_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="must not contain"):
            smoke_test._jailed_output_dir("../escape")
        assert not (tmp_path.parent / "escape").exists()

    def test_rejects_absolute_path_outside_cwd(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)
        outside = tmp_path / "outside"
        with pytest.raises(ValueError, match="must be under the current working directory"):
            smoke_test._jailed_output_dir(str(outside))
        # Containment is checked before mkdir — rejected paths must not appear.
        assert not outside.exists()

    def test_rejects_directory_symlink_escaping_cwd(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd"
        outside = tmp_path / "outside"
        cwd_dir.mkdir()
        outside.mkdir()
        monkeypatch.chdir(cwd_dir)
        link = cwd_dir / "link"
        link.symlink_to(outside)
        # resolve() follows the link so the candidate lands outside cwd and is rejected.
        with pytest.raises(ValueError, match="must be under the current working directory"):
            smoke_test._jailed_output_dir("link/output")
        assert not (outside / "output").exists()


    def test_windows_fallback_rejects_non_cwd_output_dir(self, tmp_path, monkeypatch):
        """Without dirfd, only --output-dir . is accepted."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        with pytest.raises(ValueError, match="requires openat/dirfd"):
            smoke_test._jailed_output_dir("smoke_output")
        assert not (tmp_path / "smoke_output").exists()

    def test_windows_fallback_allows_cwd_dot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        result = smoke_test._jailed_output_dir(".")
        assert result == tmp_path.resolve()

    def test_returns_new_path_object_not_the_raw_resolved_one(self, tmp_path, monkeypatch):
        # The jailed path is rebuilt from cwd + relative parts rather than
        # returned as the raw resolved candidate, so the CLI string is never
        # itself used as the returned filesystem path.
        monkeypatch.chdir(tmp_path)
        result = smoke_test._jailed_output_dir("out")
        assert result.is_relative_to(tmp_path.resolve())


class TestWriteJsonUnder:
    def test_writes_json_with_trailing_newline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = {"a": 1, "b": [1, 2, 3]}
        smoke_test._write_json_under(tmp_path, "manifest.json", data)

        path = tmp_path / "manifest.json"
        content = path.read_text(encoding="utf-8")
        assert content.endswith("\n")
        assert json.loads(content) == data

    def test_rejects_unsafe_artifact_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="plain basename"):
            smoke_test._write_json_under(tmp_path, "../escape.json", {"x": 1})
        assert not (tmp_path.parent / "escape.json").exists()

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "manifest.json"
        path.write_text("stale")
        smoke_test._write_json_under(tmp_path, "manifest.json", {"fresh": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {"fresh": True}

    def test_refuses_to_write_through_symlink(self, tmp_path, monkeypatch):
        """Writing must not follow a symlink artifact to its target."""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "elsewhere.txt"
        target.write_text("original", encoding="utf-8")
        link = tmp_path / "manifest.json"
        link.symlink_to(target)
        # Exclusive temp + replace either fails closed or replaces the symlink
        # dirent with a new regular file — never truncates the outside target.
        try:
            smoke_test._write_json_under(tmp_path, "manifest.json", {"hijacked": True})
        except (OSError, ValueError):
            pass
        assert target.read_text(encoding="utf-8") == "original"
        if link.exists() and not link.is_symlink():
            assert json.loads(link.read_text(encoding="utf-8")) == {"hijacked": True}

    def test_windows_fallback_refuses_symlink_artifact(self, tmp_path, monkeypatch):
        """Non-dirfd path must fail closed on symlink artifacts (Codex P2)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        target = tmp_path / "elsewhere.txt"
        target.write_text("original", encoding="utf-8")
        link = tmp_path / "manifest.json"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="symlink artifact"):
            smoke_test._write_json_under(tmp_path, "manifest.json", {"hijacked": True})
        assert target.read_text(encoding="utf-8") == "original"

    def test_windows_fallback_write_does_not_follow_symlink_race(self, tmp_path, monkeypatch):
        """Exclusive temp+replace must not truncate a symlink target (Codex P2)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        target = tmp_path / "elsewhere.txt"
        target.write_text("original", encoding="utf-8")
        link = tmp_path / "manifest.json"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="symlink|reparse"):
            smoke_test._write_json_under(tmp_path, "manifest.json", {"hijacked": True})
        assert target.read_text(encoding="utf-8") == "original"

    def test_windows_fallback_write_overwrites_regular_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        path = tmp_path / "manifest.json"
        path.write_text("stale", encoding="utf-8")
        smoke_test._write_json_under(tmp_path, "manifest.json", {"fresh": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {"fresh": True}
        assert not path.is_symlink()

    def test_write_does_not_truncate_hard_link_target(self, tmp_path, monkeypatch):
        """O_TRUNC on a hard-linked artifact must not clobber the outside inode."""
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / f"hardlink_target_{tmp_path.name}.txt"
        outside.write_text("outside-original", encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        artifact = out_dir / "manifest.json"
        try:
            os.link(outside, artifact)
        except OSError as exc:
            pytest.skip(f"hard links unsupported: {exc}")
        smoke_test._write_json_under(out_dir, "manifest.json", {"fresh": True})
        assert outside.read_text(encoding="utf-8") == "outside-original"
        assert json.loads(artifact.read_text(encoding="utf-8")) == {"fresh": True}
        # artifact should no longer share inode with outside
        assert os.stat(artifact).st_ino != os.stat(outside).st_ino
        outside.unlink(missing_ok=True)

    def test_windows_fallback_write_rejects_non_cwd_parent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        nested = tmp_path / "smoke_output"
        nested.mkdir()
        with pytest.raises(ValueError, match="process cwd"):
            smoke_test._write_json_under(nested, "manifest.json", {"x": 1})

    def test_replace_removes_temp_when_destination_is_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(smoke_test, "_supports_dirfd", lambda: False)
        dest = tmp_path / "manifest.json"
        dest.mkdir()  # directory occupying artifact basename
        with pytest.raises(ValueError, match="directory"):
            smoke_test._write_json_under(tmp_path, "manifest.json", {"x": 1})
        leftovers = list(tmp_path.glob(".manifest.json.*.tmp"))
        assert leftovers == []

    def test_open_path_under_cwd_uses_dot_anchor(self, tmp_path, monkeypatch):
        """cwd openat root must be process '.' not a re-resolved absolute path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()
        opened = []
        real_open = os.open

        def tracking_open(path, flags, *args, **kwargs):
            opened.append((path, flags, kwargs.get("dir_fd")))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(smoke_test.os, "open", tracking_open)
        fd = smoke_test._open_path_under_cwd(("out",), create=False)
        smoke_test.os.close(fd)
        # First open must be the process cwd anchor "."
        assert opened, "expected os.open calls"
        first_path = opened[0][0]
        assert first_path == "." or first_path == b".", f"expected '.' anchor, got {first_path!r}"

    def test_missing_out_dir_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            smoke_test._write_json_under(missing, "manifest.json", {})

    def test_rejects_out_of_cwd_parent(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd"
        outside = tmp_path / "outside"
        cwd_dir.mkdir()
        outside.mkdir()
        monkeypatch.chdir(cwd_dir)
        with pytest.raises(ValueError, match="must be under the current working directory"):
            smoke_test._write_json_under(outside, "manifest.json", {"x": 1})
        assert not (outside / "manifest.json").exists()


class TestWriteResultsJsonl:
    def test_writes_one_json_object_per_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = [
            {"request_id": 1, "status": "ok"},
            {"request_id": 2, "status": "error"},
        ]
        smoke_test._write_results_jsonl(tmp_path, results)

        path = tmp_path / "results.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == results[0]
        assert json.loads(lines[1]) == results[1]

    def test_empty_results_writes_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        smoke_test._write_results_jsonl(tmp_path, [])
        path = tmp_path / "results.jsonl"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""


class TestWriteSummaryUnder:
    def _sample_results(self):
        return [
            {"status": "ok", "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 100.0},
            {"status": "ok", "prompt_tokens": 20, "completion_tokens": 10, "latency_ms": 200.0},
            {"status": "error", "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0},
        ]

    def test_summary_contains_expected_stats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = self._sample_results()
        delta = {"tokens_prompt": 30, "tokens_completion": 15, "tokens_total": 45, "requests": 3}
        smoke_test._write_summary_under(tmp_path, results, delta, dry_run=False, model="my-model")

        text = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert text.startswith("# Python Smoke Test Summary")
        assert "`my-model`" in text
        assert "- **Mode**: live" in text
        assert "**Total requests**: 3" in text
        assert "**OK**: 2 | **Errors**: 1 | **Dry-run**: 0" in text
        assert "**Prompt tokens**: 30" in text
        assert "**Completion tokens**: 15" in text
        assert "**Avg latency**: 150.0 ms" in text
        assert "| Prompt tokens | 30 |" in text
        assert "| Completion tokens | 15 |" in text
        assert "| Total tokens | 45 |" in text
        assert "| Requests | 3 |" in text
        assert text.endswith("\n")

    def test_dry_run_mode_label_and_counts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = [{"status": "dry_run", "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}]
        delta = {"tokens_prompt": 0, "tokens_completion": 0, "tokens_total": 0, "requests": 0}
        smoke_test._write_summary_under(tmp_path, results, delta, dry_run=True, model="m")

        text = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "- **Mode**: dry-run" in text
        assert "**Dry-run**: 1" in text

    def test_no_ok_results_avg_latency_is_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = [{"status": "error", "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}]
        delta = {"tokens_prompt": 0, "tokens_completion": 0, "tokens_total": 0, "requests": 0}
        smoke_test._write_summary_under(tmp_path, results, delta, dry_run=False, model="m")

        text = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "**Avg latency**: 0.0 ms" in text

    def test_empty_results_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        delta = {"tokens_prompt": 0, "tokens_completion": 0, "tokens_total": 0, "requests": 0}
        smoke_test._write_summary_under(tmp_path, [], delta, dry_run=True, model="m")

        text = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "**Total requests**: 0" in text

    def test_missing_out_dir_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            smoke_test._write_summary_under(missing, [], {}, dry_run=True, model="m")


class TestMainIntegration:
    def test_dry_run_creates_all_artifacts_under_jailed_output_dir(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "smoke_test.py",
                "--model",
                "tiny-model",
                "--workload",
                "inference",
                "--max-requests",
                "2",
                "--concurrency",
                "1",
                "--dry-run",
                "--output-dir",
                "smoke_output",
            ],
        )

        smoke_test.main()

        out_dir = tmp_path / "smoke_output"
        for name in [
            "manifest.json",
            "provider.json",
            "usage_before.json",
            "usage_after.json",
            "usage_delta.json",
            "results.jsonl",
            "summary.md",
        ]:
            assert (out_dir / name).is_file(), f"missing artifact: {name}"

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["config"]["model_id"] == "tiny-model"
        assert manifest["config"]["dry_run"] is True

        results = [
            json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(results) == 2
        assert all(r["status"] == "dry_run" for r in results)

        captured = capsys.readouterr()
        assert "Completed 2 requests" in captured.out

    def test_output_dir_with_parent_traversal_raises_before_writing(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "smoke_test.py",
                "--model",
                "tiny-model",
                "--max-requests",
                "1",
                "--dry-run",
                "--output-dir",
                "../escape",
            ],
        )

        with pytest.raises(ValueError, match="must not contain"):
            smoke_test.main()

        assert not (tmp_path / "escape").exists()

    def test_output_dir_absolute_outside_cwd_raises(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)
        outside = tmp_path / "outside"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "smoke_test.py",
                "--model",
                "tiny-model",
                "--max-requests",
                "1",
                "--dry-run",
                "--output-dir",
                str(outside),
            ],
        )

        with pytest.raises(ValueError, match="must be under the current working directory"):
            smoke_test.main()

        # Jail validates before mkdir — rejected absolute paths must not appear.
        assert not outside.exists()
