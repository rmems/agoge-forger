from pathlib import Path


def _check_no_parent_traversal(candidate: Path) -> None:
    """Reject a candidate path that explicitly traverses '..' segments.

    Checking the *pre-resolution* path catches the common traversal
    patterns (`../etc/passwd`, `safe/../../escape`) before they reach
    the filesystem. We deliberately check the candidate instead of the
    resolved path because Python's `Path.resolve()` consumes '..'
    segments, so a post-resolve check would not see them.
    """
    if ".." in candidate.parts:
        raise ValueError(f"Path must not contain '..': {candidate}")


def _refuse_symlink_output_components(candidate: Path) -> None:
    """Refuse a destination whose leaf or any ancestor is a symlink.

    ``Path.resolve()`` follows links, so a dangling leaf or a redirected parent
    would publish an immutable snapshot somewhere other than the operator-
    specified path. Every component is checked, matching cleanup-run: stopping at
    the first existing ancestor would miss a symlink above an already-created
    directory (``link/existing/snapshot``).
    """
    probe = candidate if candidate.is_absolute() else Path.cwd() / candidate
    for component in (probe, *probe.parents):
        if component.is_symlink():
            raise ValueError(f"Refusing to publish through a symlinked path: {component}")


def resolve_existing_path(
    path: str, *, must_be_file: bool = False, must_be_dir: bool = False
) -> Path:
    if not path or not path.strip():
        raise ValueError("Path must not be empty")

    candidate = Path(path).expanduser()
    _check_no_parent_traversal(candidate)
    # strict=True preserves the distinction between a missing path and an
    # inaccessible one: the latter raises PermissionError instead of being
    # flattened into Path.exists() == False.
    resolved = candidate.resolve(strict=True)
    if must_be_file and not resolved.is_file():
        raise ValueError(f"Expected a file path: {resolved}")
    if must_be_dir and not resolved.is_dir():
        raise ValueError(f"Expected a directory path: {resolved}")
    return resolved


def resolve_output_directory(path: str) -> Path:
    if not path or not path.strip():
        raise ValueError("Output directory must not be empty")

    candidate = Path(path).expanduser()
    _check_no_parent_traversal(candidate)
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_absent_output_directory(path: str) -> Path:
    """Resolve a new output directory without creating the leaf path.

    Frozen-split publication requires the destination not to exist. Callers that
    need a created directory should keep using ``resolve_output_directory``.
    Symlinked leaves or parents are refused so publication cannot be redirected.
    """
    if not path or not path.strip():
        raise ValueError("Output directory must not be empty")

    candidate = Path(path).expanduser()
    _check_no_parent_traversal(candidate)
    _refuse_symlink_output_components(candidate)
    resolved = candidate.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
