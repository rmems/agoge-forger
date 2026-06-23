from pathlib import Path


def _reject_parent_traversal(path: Path) -> None:
    if ".." in path.parts:
        raise ValueError(f"Path must not contain '..': {path}")


def resolve_existing_path(path: str, *, must_be_file: bool = False, must_be_dir: bool = False) -> Path:
    if not path or not str(path).strip():
        raise ValueError("Path must not be empty")

    candidate = Path(path).expanduser()
    _reject_parent_traversal(candidate)
    resolved = candidate.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    if must_be_file and not resolved.is_file():
        raise ValueError(f"Expected a file path: {resolved}")
    if must_be_dir and not resolved.is_dir():
        raise ValueError(f"Expected a directory path: {resolved}")
    return resolved


def resolve_output_directory(path: str) -> Path:
    if not path or not str(path).strip():
        raise ValueError("Output directory must not be empty")

    candidate = Path(path).expanduser()
    _reject_parent_traversal(candidate)
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
