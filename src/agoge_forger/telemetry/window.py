"""How a profiling window is requested: ``run_id`` + phase + inclusive step range."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from ..config import ProfileWindowConfig, TelemetryConfig
from .schema import TORCH_BACKEND

_COMPACT = re.compile(
    r"^(?P<phase>[A-Za-z_][A-Za-z0-9_]*)[:/](?P<start>\d+)(?:-|:|\.\.)(?P<end>\d+)$"
)
_ENV_RUN_ID = "AGOGE_RUN_ID"
_ENV_WINDOW = "AGOGE_PROFILE_WINDOW"
_ENV_BACKEND = "AGOGE_PROFILE_BACKEND"


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
    """YAML first; env can fill a missing run id and enable a window that YAML left off."""

    run_id = telemetry.run_id or _nonempty(environ.get(_ENV_RUN_ID))
    window = telemetry.profile_window
    env_window = _nonempty(environ.get(_ENV_WINDOW))
    if not window.enabled and env_window is not None:
        window = parse_profile_window(env_window)
    backend = _nonempty(environ.get(_ENV_BACKEND))
    if backend is not None:
        window = window.model_copy(update={"backend": backend})
    return telemetry.model_copy(update={"run_id": run_id, "profile_window": window})


def overlay_cli_telemetry(
    telemetry: TelemetryConfig,
    run_id: str | None,
    profile_window: str | None,
) -> TelemetryConfig:
    updated = overlay_telemetry_env(telemetry, os.environ)
    if run_id:
        updated = updated.model_copy(update={"run_id": run_id.strip()})
    if profile_window:
        updated = updated.model_copy(
            update={"profile_window": parse_profile_window(profile_window)}
        )
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
        start_step=int(matched.group("start")),
        end_step=int(matched.group("end")),
        backend=TORCH_BACKEND,
    )


def _parse_kv_window(raw: str) -> ProfileWindowConfig:
    fields: dict[str, Any] = {"enabled": True, "backend": TORCH_BACKEND}
    for part in raw.split(","):
        key, sep, value = part.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"invalid profile window field: {part!r}")
        fields[_normalize_kv_key(key.strip())] = value.strip()
    start_step = int(fields["start_step"]) if "start_step" in fields else 1
    end_step = int(fields["end_step"]) if "end_step" in fields else start_step
    return ProfileWindowConfig.model_validate(
        {
            "enabled": True,
            "phase": fields.get("phase", "train"),
            "start_step": start_step,
            "end_step": end_step,
            "backend": fields.get("backend", TORCH_BACKEND),
        }
    )


def _normalize_kv_key(key: str) -> str:
    aliases = {
        "start": "start_step",
        "end": "end_step",
        "from": "start_step",
        "to": "end_step",
    }
    return aliases.get(key, key)
