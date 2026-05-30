from __future__ import annotations

import pytest

from hermes_agentic_rl.trainers.on_policy import OnPolicyTrainerConfig
from hermes_agentic_rl.trainers.stability import (
    apply_stability_preset_to_config,
    expand_stability_preset,
)


def test_expand_stability_preset_respects_explicit_yaml_keys() -> None:
    expanded = expand_stability_preset(
        {
            "stability_preset": "aggressive",
            "target_kl": 0.2,
            "entropy_schedule": {"enabled": False},
        },
        explicit_keys={"stability_preset", "target_kl", "entropy_schedule"},
    )

    assert expanded["normalize_reward"] is True
    assert expanded["reward_norm_clip"] == 3.0
    assert expanded["target_kl"] == 0.2
    assert expanded["adaptive_kl"] is True
    assert expanded["entropy_schedule"] == {"enabled": False}


def test_apply_stability_preset_to_python_config_preserves_non_default_values() -> None:
    cfg = OnPolicyTrainerConfig(
        stability_preset="standard",
        target_kl=0.2,
        entropy_schedule={"mode": "cosine", "peak": 0.02},
    )

    applied = apply_stability_preset_to_config(cfg)

    assert applied.stability_preset == "standard"
    assert applied.normalize_reward is True
    assert applied.reward_norm_clip == 5.0
    assert applied.target_kl == 0.2
    assert applied.adaptive_kl is False
    assert applied.entropy_schedule == {"mode": "cosine", "peak": 0.02}


def test_unknown_stability_preset_raises() -> None:
    with pytest.raises(ValueError, match="stability_preset"):
        expand_stability_preset({"stability_preset": "rocket"})
