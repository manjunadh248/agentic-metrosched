"""Configuration loading and validation.

The YAML file under ``data/config/`` is the single source of truth for every
infrastructure and operating parameter. This module loads it, converts clock
strings ("HH:MM") into seconds-from-midnight and validates bounds so that bad
configurations fail fast with a clear :class:`ConfigError`.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "metro_config.yaml"

FLEET_MIN, FLEET_MAX = 20, 30
REQUIRED_STATES = ["Available", "Maintenance", "Inducted", "Running", "Delayed"]
SCENARIO_TYPES = ["normal", "peak_demand", "disruption"]


def clock_to_seconds(value: str | int) -> int:
    """Convert ``"HH:MM"`` or ``"HH:MM:SS"`` to seconds from midnight.

    Integers are passed through unchanged so values may also be given in seconds.
    """
    if isinstance(value, int):
        return value
    try:
        parts = [int(p) for p in str(value).split(":")]
    except ValueError as exc:
        raise ConfigError(f"Invalid clock value {value!r}") from exc
    if len(parts) == 2:
        parts.append(0)
    if len(parts) != 3:
        raise ConfigError(f"Invalid clock value {value!r}")
    h, m, s = parts
    if not (0 <= h <= 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ConfigError(f"Clock value out of range: {value!r}")
    return h * 3600 + m * 60 + s


def seconds_to_clock(seconds: float) -> str:
    """Format seconds-from-midnight as ``HH:MM:SS``."""
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _require(cfg: dict, path: str) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Missing required config key: {path}")
        node = node[key]
    return node


def _check_range(name: str, pair: Any) -> None:
    if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] <= pair[1] and pair[0] >= 0):
        raise ConfigError(f"{name} must be a [min, max] pair with 0 <= min <= max, got {pair!r}")


def _normalise(cfg: dict) -> dict:
    """Convert clock strings to seconds in-place on a copy."""
    cfg = copy.deepcopy(cfg)
    sim = cfg["simulation"]
    for key in ("depot_open", "service_start", "service_end"):
        sim[key] = clock_to_seconds(sim[key])
    for w in cfg["depot"]["induction_windows"]:
        w["start"] = clock_to_seconds(w["start"])
        w["end"] = clock_to_seconds(w["end"])
    for p in cfg["demand"]["peaks"]:
        p["center"] = clock_to_seconds(p["center"])
    return cfg


def validate_config(cfg: dict) -> None:
    """Validate a *normalised* configuration dictionary. Raises ConfigError."""
    sim = cfg["simulation"]
    if not sim["depot_open"] <= sim["service_start"] < sim["service_end"] <= 24 * 3600:
        raise ConfigError("Require depot_open <= service_start < service_end <= 24:00")
    for key in ("interval_s", "decision_interval_s"):
        if int(sim[key]) <= 0:
            raise ConfigError(f"simulation.{key} must be positive")

    fleet = cfg["fleet"]
    size = int(fleet["size"])
    if not FLEET_MIN <= size <= FLEET_MAX:
        raise ConfigError(f"fleet.size must be within {FLEET_MIN}..{FLEET_MAX}, got {size}")
    if list(fleet["states"]) != REQUIRED_STATES:
        raise ConfigError(f"fleet.states must be exactly {REQUIRED_STATES}")
    _check_range("fleet.min_prep_time_s", fleet["min_prep_time_s"])
    _check_range("fleet.km_since_maintenance", fleet["km_since_maintenance"])
    if fleet["train_capacity"] <= 0 or fleet["priority_levels"] < 1:
        raise ConfigError("fleet.train_capacity and fleet.priority_levels must be positive")

    depot = cfg["depot"]
    if depot["stabling_lines"] * depot["berths_per_line"] < size:
        raise ConfigError("Depot stabling capacity is smaller than the fleet size")
    if not depot["exit_tracks"]:
        raise ConfigError("At least one depot exit track is required")
    if depot["max_inductions_per_hour"] < 1:
        raise ConfigError("depot.max_inductions_per_hour must be >= 1")
    windows = sorted(depot["induction_windows"], key=lambda w: w["start"])
    for w in windows:
        if w["start"] >= w["end"] or w["max_inductions"] < 0:
            raise ConfigError(f"Invalid induction window {w['id']}")
    for a, b in zip(windows, windows[1:]):
        if a["end"] > b["start"]:
            raise ConfigError(f"Induction windows {a['id']} and {b['id']} overlap")

    sections = (
        list(depot["exit_tracks"])
        + [depot["reception_track"]]
        + list(cfg["track"]["shared_sections"])
        + list(cfg["track"]["entry_points"])
    )
    ids = [s["id"] for s in sections]
    if len(ids) != len(set(ids)):
        raise ConfigError("Track section IDs must be unique")
    for s in sections:
        if s["traversal_s"] <= 0 or s["headway_s"] < 0:
            raise ConfigError(f"Section {s['id']} needs traversal_s > 0 and headway_s >= 0")
    if not cfg["track"]["entry_points"]:
        raise ConfigError("At least one mainline entry point is required")

    if cfg["platforms"]["count"] < 1:
        raise ConfigError("platforms.count must be >= 1")

    svc = cfg["service"]
    if not 0 < svc["min_service_headway_s"] <= svc["max_service_headway_s"]:
        raise ConfigError("Require 0 < min_service_headway_s <= max_service_headway_s")
    if not 0 < svc["target_load_factor"] <= 1.5:
        raise ConfigError("service.target_load_factor must be in (0, 1.5]")
    if not 0 <= svc["reserve_trains"] < size:
        raise ConfigError("service.reserve_trains must be in [0, fleet.size)")

    dem = cfg["demand"]
    if dem["base_pphpd"] <= 0:
        raise ConfigError("demand.base_pphpd must be positive")
    if len(dem["weekday_factors"]) != 7:
        raise ConfigError("demand.weekday_factors must have 7 entries (Mon..Sun)")

    ops = cfg["operations"]
    for key in ("trip_delay_prob", "in_service_failure_prob"):
        if not 0 <= ops[key] <= 1:
            raise ConfigError(f"operations.{key} must be a probability")
    _check_range("operations.repair_time_s", ops["repair_time_s"])

    scen = cfg["scenarios"]
    mix = scen["mix"]
    if sorted(mix) != sorted(SCENARIO_TYPES):
        raise ConfigError(f"scenarios.mix must define exactly {SCENARIO_TYPES}")
    if abs(sum(mix.values()) - 1.0) > 1e-6 or min(mix.values()) < 0:
        raise ConfigError("scenarios.mix probabilities must be >= 0 and sum to 1")
    for st in SCENARIO_TYPES:
        params = scen.get(st)
        if not isinstance(params, dict):
            raise ConfigError(f"scenarios.{st} parameters missing")
        for key in ("prep_delay_prob", "depot_fault_prob"):
            if not 0 <= params[key] <= 1:
                raise ConfigError(f"scenarios.{st}.{key} must be a probability")
        if params["exit_track_closures"] >= len(depot["exit_tracks"]):
            raise ConfigError(f"scenarios.{st}: cannot close every exit track")
        if params["platform_closures"] >= cfg["platforms"]["count"]:
            raise ConfigError(f"scenarios.{st}: cannot close every platform")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load, normalise and validate the YAML configuration.

    Args:
        path: YAML file path; defaults to ``data/config/metro_config.yaml``.
        overrides: optional nested dict merged on top of the file (used by
            tests and, later, the dashboard). Clock values in overrides may be
            "HH:MM" strings.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    if overrides:
        raw = deep_merge(raw, overrides)
    try:
        for key in ("simulation", "fleet", "depot", "track", "platforms", "service", "demand",
                    "operations", "scenarios"):
            _require(raw, key)
        cfg = _normalise(raw)
        validate_config(cfg)
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Malformed configuration: {exc!r}") from exc
    logger.debug("Loaded configuration from %s (fleet=%s)", path, cfg["fleet"]["size"])
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` recursively merged into ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
