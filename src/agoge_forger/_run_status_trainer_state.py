"""Non-executing Trainer-state integrity checks used by run-status."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

_RANKED_RNG_RE = re.compile(r"rng_state_(\d+)\.pth")


def _json_object(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return isinstance(payload, dict)


def _torch_archive_has_records(names: list[str]) -> bool:
    roots = {name.partition("/")[0] for name in names}
    if len(roots) != 1:
        return False
    root = roots.pop()
    required = {f"{root}/data.pkl", f"{root}/version", f"{root}/.data/serialization_id"}
    return required.issubset(names)


def _torch_state_usable(path: Path) -> bool:
    """Validate current PyTorch's ZIP records and CRCs without unpickling."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle, zipfile.ZipFile(handle) as archive:
            return _torch_archive_has_records(archive.namelist()) and archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _rng_rank(path: Path) -> int | None:
    match = _RANKED_RNG_RE.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def _ranked_rng_states_usable(paths: list[Path]) -> bool:
    ranks = [_rng_rank(path) for path in paths]
    if any(rank is None for rank in ranks):
        return False
    return set(ranks) == set(range(len(paths))) and all(_torch_state_usable(path) for path in paths)


def _rng_state_usable(checkpoint: Path) -> bool:
    single = checkpoint / "rng_state.pth"
    if single.is_file():
        return _torch_state_usable(single)
    ranked = sorted(checkpoint.glob("rng_state_*.pth"))
    return bool(ranked) and _ranked_rng_states_usable(ranked)


def trainer_state_usable(checkpoint: str | Path | None) -> bool:
    if checkpoint is None:
        return False
    checkpoint_dir = Path(checkpoint)
    return bool(
        _json_object(checkpoint_dir / "trainer_state.json")
        and _torch_state_usable(checkpoint_dir / "optimizer.pt")
        and _torch_state_usable(checkpoint_dir / "scheduler.pt")
        and _rng_state_usable(checkpoint_dir)
    )
