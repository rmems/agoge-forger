"""How a profiling window is requested: ``run_id`` + phase + inclusive step range."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from ..config import ProfileWindowConfig, TelemetryConfig
from .schema import TORCH_BACKEND, TRAIN_PHASE

_COMPACT = re.compile(
    r"^(?P<phase>[A-Za-z_][A-Za-z0-9_]*)[:/](?P<start>\d+)(?:-|:|\.\.)(?P<end>\d+)$"
)
_ENV_RUN_ID = "AGOGE_RUN_ID"
_ENV_WINDOW = "AGOGE_PROFILE_WINDOW"
_ENV_BACKEND = "AGOGE_PROFILE_BACKEND"
_ALLOWED_KV = frozenset({"phase", "start_step", "end_step", "backend", "enabled"})
_KV_ALIASES = {
    "start": "start_step",
    "end": "end_step",
    "from": "start_step",
    "to": "end_step",
}


def parse_profile_window(raw: str) -> ProfileWindowConfig:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("profile window request is empty")
    if "=" in stripped:
        return _parse_kv_window(stripped)
    return _parse_compact_window(stripped)


def profile_window_id(run_id: str, window: ProfileWindowConfig) -> str:
    return f"{run_id}:{window.phase}:{window.start_step}-{window.end_step}"


def window_covers_step(window: ProfileWindowConfig, step: int) -> bool:
    return window.enabled and window.start_step <= step <= window.end_step


def upcoming_step_starts_window(window: ProfileWindowConfig, upcoming: int) -> bool:
    return window.enabled and upcoming == window.start_step


def overlay_telemetry_env(
    telemetry: TelemetryConfig, environ: Mapping[str, str]
) -> TelemetryConfig:
    """Apply environment values over YAML/default telemetry exactly once."""

    run_id = _nonempty(environ.get(_ENV_RUN_ID)) or telemetry.run_id
    window = telemetry.profile_window
    env_window = _nonempty(environ.get(_ENV_WINDOW))
    if env_window is not None:
        window = parse_profile_window(env_window)
    backend = _nonempty(environ.get(_ENV_BACKEND))
    if backend is not None:
        window = window.model_copy(update={"backend": backend})
    updated = telemetry.model_copy(
        update={
            "run_id": run_id,
            "profile_window": window,
        }
    )
    updated._environment_resolved = True
    return updated


def overlay_cli_telemetry(
    telemetry: TelemetryConfig,
    run_id: str | None,
    profile_window: str | None,
) -> TelemetryConfig:
    environ = dict(os.environ)
    if profile_window:
        environ.pop(_ENV_WINDOW, None)
    updated = overlay_telemetry_env(telemetry, environ)
    if run_id:
        updated = updated.model_copy(update={"run_id": run_id.strip()})
    if profile_window:
        updated = updated.model_copy(
            update={"profile_window": parse_profile_window(profile_window)}
        )
    updated._environment_resolved = True
    return updated


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_compact_window(raw: str) -> ProfileWindowConfig:
    matched = _COMPACT.fullmatch(raw)
    if matched is None:
        raise ValueError(
            "profile window must look like 'train:1-2' or 'phase=train,start_step=1,end_step=2'"
        )
    return ProfileWindowConfig(
        enabled=True,
        phase=matched.group("phase"),
        start_step=_require_int(matched.group("start"), "start"),
        end_step=_require_int(matched.group("end"), "end"),
        backend=TORCH_BACKEND,
    )


def _parse_kv_window(raw: str) -> ProfileWindowConfig:
    fields = _kv_fields(raw)
    start_step = _optional_int(fields, "start_step", 1)
    end_step = _optional_int(fields, "end_step", start_step)
    return ProfileWindowConfig.model_validate(
        {
            "enabled": _optional_bool(fields, "enabled", True),
            "phase": fields.get("phase", TRAIN_PHASE),
            "start_step": start_step,
            "end_step": end_step,
            "backend": fields.get("backend", TORCH_BACKEND),
        }
    )


def _kv_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in raw.split(","):
        key, value = _kv_part(part)
        if key in fields:
            raise ValueError(f"duplicate profile window field: {key}")
        if key not in _ALLOWED_KV:
            raise ValueError(f"unknown profile window field: {key}")
        fields[key] = value
    return fields


def _kv_part(part: str) -> tuple[str, str]:
    key, sep, value = part.partition("=")
    if not _valid_kv_part(sep, key, value):
        raise ValueError(f"invalid profile window field: {part!r}")
    return _normalize_kv_key(key.strip()), value.strip()


def _valid_kv_part(sep: str, key: str, value: str) -> bool:
    if not sep:
        return False
    if not key.strip():
        return False
    return bool(value.strip())


def _normalize_kv_key(key: str) -> str:
    return _KV_ALIASES.get(key, key)


def _optional_int(fields: Mapping[str, str], key: str, default: int) -> int:
    if key not in fields:
        return default
    return _require_int(fields[key], key)


def _require_int(raw: str, label: str) -> int:
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"profile window {label} must be an integer") from error


def _optional_bool(fields: Mapping[str, str], key: str, default: bool) -> bool:
    if key not in fields:
        return default
    normalized = fields[key].casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"profile window {key} must be true or false")
