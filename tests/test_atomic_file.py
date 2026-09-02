import pytest

from agoge_forger import _atomic_file


def test_publication_probe_fails_before_payload_writer(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.json"
    writer_called = False

    def unsupported(staging_parent):
        assert staging_parent == tmp_path
        raise OSError("atomic publication unsupported")

    def writer(path, payload):
        nonlocal writer_called
        writer_called = True

    monkeypatch.setattr(_atomic_file, "require_rename_noreplace_support", unsupported)

    with pytest.raises(OSError, match="atomic publication unsupported"):
        _atomic_file.publish_bytes_noreplace(
            destination,
            b"complete",
            refusal="must not overwrite",
            writer=writer,
        )

    assert not writer_called
    assert not destination.exists()
