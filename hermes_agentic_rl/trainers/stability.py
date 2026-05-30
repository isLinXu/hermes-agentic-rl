"""Stability presets for on-policy trainers.

The individual knobs remain first-class config fields. Presets are a thin
expansion layer for common, internally consistent combinations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from typing import Any, Literal, TypeVar

StabilityPreset = Literal["none", "standard", "aggressive"]
ConfigT = TypeVar("ConfigT")

STABILITY_PRESETS: dict[StabilityPreset, dict[str, Any]] = {
    "none": {},
    "standard": {
        "normalize_reward": True,
        "reward_norm_clip": 5.0,
        "target_kl": 0.1,
        "adaptive_kl": False,
        "entropy_schedule": {
            "mode": "linear",
            "start": 0.01,
            "end": 0.001,
        },
    },
    "aggressive": {
        "normalize_reward": True,
        "reward_norm_clip": 3.0,
        "target_kl": 0.05,
        "adaptive_kl": True,
        "adaptive_kl_horizon": 5000.0,
        "entropy_schedule": {
            "mode": "pid",
            "target_entropy": 1.5,
        },
    },
}


def normalize_stability_preset(value: Any) -> StabilityPreset:
    """Validate and normalize a preset name."""
    preset = str(value or "none").lower()
    if preset in STABILITY_PRESETS:
        return preset  # type: ignore[return-value]
    choices = ", ".join(STABILITY_PRESETS)
    raise ValueError(f"stability_preset must be one of: {choices}")


def expand_stability_preset(
    values: dict[str, Any],
    *,
    explicit_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Return config values with a stability preset expanded.

    When ``explicit_keys`` is provided, those keys win over preset defaults.
    This lets YAML users opt into a preset while overriding one field.
    """
    result = dict(values)
    preset = normalize_stability_preset(result.get("stability_preset", "none"))
    result["stability_preset"] = preset
    if preset == "none":
        return result

    explicit = explicit_keys or frozenset()
    for key, preset_value in STABILITY_PRESETS[preset].items():
        if key in explicit:
            continue
        result[key] = deepcopy(preset_value)
    return result


def apply_stability_preset_to_config(cfg: ConfigT) -> ConfigT:
    """Apply a trainer config preset while preserving non-default overrides.

    Dataclass configs created directly from Python do not carry "explicit key"
    metadata, so this treats values different from the dataclass default as
    user overrides.
    """
    if not is_dataclass(cfg):
        return cfg

    field_by_name = {field.name: field for field in fields(cfg)}
    preset_field = field_by_name.get("stability_preset")
    if preset_field is None:
        return cfg

    preset = normalize_stability_preset(getattr(cfg, "stability_preset", "none"))
    if preset == "none":
        return cfg

    updates: dict[str, Any] = {"stability_preset": preset}
    for key, preset_value in STABILITY_PRESETS[preset].items():
        field = field_by_name.get(key)
        if field is None:
            continue
        default = field.default
        if getattr(cfg, key) == default:
            updates[key] = deepcopy(preset_value)
    return replace(cfg, **updates)
