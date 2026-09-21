#!/usr/bin/env python3
"""Validate JSONL trajectory files against schemas.

Usage:
    python scripts/validate_jsonl.py datasets/jsonl/*.jsonl
    python scripts/validate_jsonl.py --strict-policy datasets/jsonl/*.jsonl

Each file is read line-by-line. Each non-empty line must be valid JSON
conforming to the PR Trajectory schema. Exits 0 if all records pass,
non-zero if any validation error is found.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from urllib.parse import urlparse
import json
import re
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None
    print(
        "ERROR: jsonschema is required. Install with: pip install jsonschema",
        file=sys.stderr,
    )
    sys.exit(2)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.secrets import find_secrets  # noqa: E402

SCHEMA_V0_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "pr_trajectory.schema.json"
)
SCHEMA_V1_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "trajectory_v1.schema.json"
)
SCHEMA_V1_1_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "trajectory_v1_1.schema.json"
)
_V1_VERSIONS = frozenset({"1", "1.0", "v1"})
_V1_1_VERSIONS = frozenset({"1.1", "v1.1"})
HOME_PATH_RE = re.compile(
    r"("
    r"/home/[A-Za-z0-9._-]+"
    r"|/Users/[A-Za-z0-9._-]+"
    r"|/root(?:/[^\s\"']+)?"
    r"|(?:[A-Za-z]:\\Users\\|[A-Za-z]:/Users/)[A-Za-z0-9._-]+"
    r")",
    re.IGNORECASE,
)
_GIT_OID_RE = re.compile(r"^[0-9a-fA-F]{3,64}$")
_SNAPSHOT_KEYS = (
    "before_blob",
    "after_blob",
    "base_oid",
    "commit_oid",
    "tree_oid",
    "head_oid",
)
_AUTHORITY_SCHEMES = frozenset({"http", "https", "ftp", "ftps"})


def load_schema(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: Schema not found at {path}", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        return json.load(f)


def _iter_strings(obj: object):
    """Yield raw string keys and values from a nested JSON structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_strings(value)


def _is_absolute_uri(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    if not parsed.scheme:
        return False
    if parsed.scheme.lower() in _AUTHORITY_SCHEMES:
        return bool(parsed.netloc)
    return True


def _is_git_oid(value: object) -> bool:
    return isinstance(value, str) and bool(_GIT_OID_RE.fullmatch(value))


def _code_state_has_git_oid(code_state: object) -> bool:
    if not isinstance(code_state, dict):
        return False
    return any(_is_git_oid(code_state.get(key)) for key in _SNAPSHOT_KEYS)


def _event_has_evidence_url(event: dict) -> bool:
    refs = event.get("evidence_references")
    if not isinstance(refs, list):
        return False
    return any(_is_absolute_uri(ref) for ref in refs)


def _event_has_auditable_anchor(event: dict) -> bool:
    return _event_has_evidence_url(event) or _code_state_has_git_oid(
        event.get("code_state")
    )


def _contains_nonfinite(obj: object) -> bool:
    if isinstance(obj, float) and not math.isfinite(obj):
        return True
    if isinstance(obj, dict):
        return any(_contains_nonfinite(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_nonfinite(v) for v in obj)
    return False


def policy_errors(record: dict, lineno: int, filename: str) -> list[str]:
    """Extra policy checks beyond JSON Schema."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return errors

    schema_version = record.get("schema_version")
    if schema_version in _V1_VERSIONS or schema_version in _V1_1_VERSIONS:
        events = record.get("events")
        if isinstance(events, list):
            last_dt: datetime | None = None
            for e in events:
                if not isinstance(e, dict):
                    continue
                ts = e.get("timestamp")
                if isinstance(ts, str) and ts:
                    try:
                        iso_ts = ts[:-1] + "+00:00" if ts.endswith(("Z", "z")) else ts
                        dt = datetime.fromisoformat(iso_ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        if last_dt is not None and dt < last_dt:
                            errors.append(
                                f"  {filename}:{lineno} [policy] - future-event leakage / events not ordered "
                                f"(timestamp {ts} before previous)"
                            )
                        last_dt = dt
                    except OverflowError:
                        errors.append(
                            f"  {filename}:{lineno} [policy] - timestamp UTC normalization overflow"
                        )
                    except (ValueError, TypeError):
                        errors.append(
                            f"  {filename}:{lineno} [policy] - timestamp is not a parseable UTC instant"
                        )

                actor = e.get("actor")
                if isinstance(actor, dict):
                    if actor.get("type") not in (
                        "human",
                        "bot",
                        "application",
                        "agent",
                    ):
                        errors.append(
                            f"  {filename}:{lineno} [policy] - invented/unsupported actor type"
                        )

                if not _event_has_auditable_anchor(e):
                    errors.append(
                        f"  {filename}:{lineno} [policy] - event missing auditable evidence anchor "
                        f"(evidence_references URL or code_state git object id)"
                    )

        traj_type = record.get("trajectory_type")
        if traj_type == "software" and isinstance(events, list):
            has_snapshot = False
            for e in events:
                if not isinstance(e, dict):
                    continue
                code_state = e.get("code_state")
                if not isinstance(code_state, dict):
                    continue
                for key in _SNAPSHOT_KEYS:
                    value = code_state.get(key)
                    if not value:
                        continue
                    if _is_git_oid(value):
                        has_snapshot = True
                    else:
                        errors.append(
                            f"  {filename}:{lineno} [policy] - code snapshot {key} is not a git object id"
                        )
            if not has_snapshot:
                errors.append(
                    f"  {filename}:{lineno} [policy] - missing required code snapshots for software trajectory"
                )

        disp = record.get("terminal_disposition")
        payload = (
            record.get("software_payload")
            if traj_type == "software"
            else record.get("research_payload")
        )
        success_dispositions = ("successful", "passed")
        success_outcomes = ("pass", "passed", "success", "successful", "verified", "ok")
        terminal_enum = {
            "successful",
            "failed",
            "reverted",
            "falsified",
            "null",
            "invalid",
            "interrupted",
            "inconclusive",
        }
        last_disp = None
        if isinstance(events, list):
            for e in reversed(events):
                if isinstance(e, dict) and "disposition" in e:
                    last_disp = e.get("disposition")
                    break
        outcome = ""
        if isinstance(payload, dict):
            outcome = str(payload.get("validation_outcome", "")).strip().lower()
        if last_disp in success_dispositions:
            terminal_success = True
        elif last_disp in (None, "neutral", "null"):
            terminal_success = (outcome in success_outcomes) if outcome else None
        else:
            terminal_success = False
        if disp == "successful" and terminal_success is False:
            errors.append(
                f"  {filename}:{lineno} [policy] - nonterminal record incorrectly represented as positive terminal example"
            )
        elif disp not in (None, "successful") and terminal_success is True:
            errors.append(
                f"  {filename}:{lineno} [policy] - terminal_disposition does not agree with terminal outcome evidence"
            )
        elif (
            isinstance(last_disp, str)
            and last_disp not in ("neutral", "null")
            and disp in terminal_enum
        ):
            normalized_last = "successful" if last_disp == "passed" else last_disp
            if normalized_last in terminal_enum and normalized_last != disp:
                errors.append(
                    f"  {filename}:{lineno} [policy] - terminal_disposition does not agree with terminal outcome evidence"
                )

        artifacts = record.get("artifacts")
        if isinstance(artifacts, list):
            for art in artifacts:
                if not isinstance(art, dict):
                    continue
                availability = art.get("availability")
                content = art.get("content")
                if availability == "inline" and not isinstance(content, str):
                    errors.append(
                        f"  {filename}:{lineno} [policy] - inline artifact missing content"
                    )
                    continue
                if availability == "remote" and not _is_absolute_uri(art.get("uri")):
                    errors.append(
                        f"  {filename}:{lineno} [policy] - remote artifact uri is not an absolute URI"
                    )
                if not isinstance(content, str):
                    continue
                try:
                    raw = content.encode("utf-8")
                except UnicodeEncodeError:
                    errors.append(
                        f"  {filename}:{lineno} [policy] - {'inline artifact' if availability == 'inline' else 'artifact'} content is not UTF-8 encodable"
                    )
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                declared = str(art.get("sha256") or "").strip().lower()
                label = "inline artifact" if availability == "inline" else "artifact"
                if declared != digest:
                    errors.append(
                        f"  {filename}:{lineno} [policy] - {label} sha256 does not match content"
                    )
                if art.get("byte_size") != len(raw):
                    errors.append(
                        f"  {filename}:{lineno} [policy] - {label} byte_size does not match content"
                    )

    repo = record.get("repo")
    pr = record.get("pr_number")
    urls = record.get("source_urls") or []
    if repo and pr:
        canonical = f"https://github.com/{repo}/pull/{pr}"
        if canonical not in urls:
            errors.append(
                f"  {filename}:{lineno} [source_urls] - missing canonical PR URL {canonical}"
            )
    # Scan raw strings so Windows paths (C:\\Users\\...) are not missed via json.dumps escapes.
    home_hit = False
    secret_families: list[str] = []
    seen_families: set[str] = set()
    for text in _iter_strings(record):
        if not home_hit and HOME_PATH_RE.search(text):
            home_hit = True
        for family in find_secrets(text):
            if family not in seen_families:
                seen_families.add(family)
                secret_families.append(family)
    if home_hit:
        errors.append(
            f"  {filename}:{lineno} [policy] - absolute user-home path present "
            f"(/home, /Users, /root, or Windows Users)"
        )
    if secret_families:
        families = ", ".join(secret_families)
        errors.append(
            f"  {filename}:{lineno} [policy] - secret-like token pattern present "
            f"({families})"
        )
    return errors


def _select_validator(
    record: object,
    v0_validator: jsonschema.Draft7Validator,
    v1_validator: jsonschema.Draft7Validator,
    v1_1_validator: jsonschema.Draft7Validator | None,
) -> jsonschema.Draft7Validator:
    if not isinstance(record, dict):
        return v0_validator
    version = record.get("schema_version")
    if version in _V1_1_VERSIONS:
        if v1_1_validator is None:
            v1_1_validator = jsonschema.Draft7Validator(
                load_schema(SCHEMA_V1_1_PATH),
                format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
            )
        return v1_1_validator
    if version in _V1_VERSIONS:
        return v1_validator
    return v0_validator


def validate_file(
    filepath: Path,
    v0_validator: jsonschema.Draft7Validator,
    v1_validator: jsonschema.Draft7Validator,
    *,
    strict_policy: bool = False,
    v1_1_validator: jsonschema.Draft7Validator | None = None,
) -> list[str]:
    """Validate a single JSONL file. Returns list of error strings."""
    errors: list[str] = []
    count = 0
    try:
        with open(filepath) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:

                    def _reject_nonfinite(constant: str):
                        raise json.JSONDecodeError(
                            f"non-finite constant {constant!r}", line, 0
                        )

                    record = json.loads(line, parse_constant=_reject_nonfinite)
                except json.JSONDecodeError as exc:
                    errors.append(f"  {filepath.name}:{lineno} - Invalid JSON: {exc}")
                    continue
                if _contains_nonfinite(record):
                    errors.append(
                        f"  {filepath.name}:{lineno} - Invalid JSON: non-finite number"
                    )
                    continue

                validator = _select_validator(
                    record, v0_validator, v1_validator, v1_1_validator
                )

                for error in sorted(
                    validator.iter_errors(record), key=lambda e: list(e.path)
                ):
                    path = ".".join(str(p) for p in error.absolute_path) or "(root)"
                    errors.append(
                        f"  {filepath.name}:{lineno} [{path}] - {error.message}"
                    )
                if strict_policy and isinstance(record, dict):
                    errors.extend(policy_errors(record, lineno, filepath.name))
    except FileNotFoundError:
        errors.append(f"  ERROR: File not found: {filepath}")
        return errors
    if count == 0:
        errors.append(f"  {filepath.name} - no non-empty JSONL records")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="JSONL files to validate")
    parser.add_argument(
        "--strict-policy",
        action="store_true",
        help="Also enforce data-policy hygiene (canonical URL, no /home/ paths, no secret hints)",
    )
    args = parser.parse_args(argv)

    schema_v0 = load_schema(SCHEMA_V0_PATH)
    v0_validator = jsonschema.Draft7Validator(
        schema_v0, format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER
    )

    schema_v1 = load_schema(SCHEMA_V1_PATH)
    v1_validator = jsonschema.Draft7Validator(
        schema_v1, format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER
    )
    v1_1_validator = None
    if SCHEMA_V1_1_PATH.exists():
        v1_1_validator = jsonschema.Draft7Validator(
            load_schema(SCHEMA_V1_1_PATH),
            format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
        )

    all_errors: list[str] = []

    for arg in args.files:
        filepath = Path(arg)
        file_errors = validate_file(
            filepath,
            v0_validator,
            v1_validator,
            strict_policy=args.strict_policy,
            v1_1_validator=v1_1_validator,
        )
        all_errors.extend(file_errors)
        if not file_errors:
            print(f"  ✓ {filepath.name}")
        else:
            print(f"  ✗ {filepath.name} ({len(file_errors)} error(s))")

    if all_errors:
        print("\nValidation FAILED:")
        for err in all_errors:
            print(err)
        return 1

    print("\nAll files passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
