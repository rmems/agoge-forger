from pathlib import Path

import pytest

from agoge_forger._atomic_directory import rename_noreplace


def test_rename_noreplace_rejects_embedded_nul_without_truncating(tmp_path: Path) -> None:
    source = tmp_path / "staged"
    source.mkdir()
    (source / "payload.txt").write_text("complete\n", encoding="utf-8")
    truncated_destination = tmp_path / "unexpected"
    destination = Path(f"{truncated_destination}\0ignored")

    with pytest.raises(ValueError, match="embedded NUL"):
        rename_noreplace(source, destination)

    assert source.is_dir()
    assert not truncated_destination.exists()
