"""Tests for GSPO/PPO algorithm routing in YAML config."""

from __future__ import annotations

from hermes_agentic_rl.yaml_config import _build_trainer_config_typed


def test_gspo_config_built_from_yaml():
    """_build_trainer_config_typed should build GSPOTrainerConfig from dict."""
    from hermes_agentic_rl.trainers.gspo_trainer import GSPOTrainerConfig

    config = {
        "algo": "gspo",
        "n_iters": 10,
        "group_size": 4,
        "clip_eps": 0.2,
        "log_ratio_clip": 30.0,
        "backend": {"type": "tiny"},
        "env": {"type": "echo"},
    }
    cfg = _build_trainer_config_typed(config, GSPOTrainerConfig)

    assert isinstance(cfg, GSPOTrainerConfig)
    assert cfg.n_iters == 10
    assert cfg.group_size == 4
    assert cfg.clip_eps == 0.2
    assert cfg.log_ratio_clip == 30.0


def test_gspo_config_ignores_unknown_keys():
    """Unknown keys should be silently ignored."""
    from hermes_agentic_rl.trainers.gspo_trainer import GSPOTrainerConfig

    config = {
        "algo": "gspo",
        "n_iters": 5,
        "unknown_field": "ignored",
        "rewards": {"components": []},
    }
    cfg = _build_trainer_config_typed(config, GSPOTrainerConfig)
    assert cfg.n_iters == 5


def test_gspo_config_skips_algo_key():
    """The 'algo' key itself should not be passed to the config dataclass."""
    from hermes_agentic_rl.trainers.gspo_trainer import GSPOTrainerConfig

    config = {"algo": "gspo", "n_iters": 3}
    cfg = _build_trainer_config_typed(config, GSPOTrainerConfig)
    assert cfg.n_iters == 3
    # GSPOTrainerConfig has no 'algo' field, so no error means it was skipped


def test_grpo_config_still_works():
    """Original build_trainer_config should still work for GRPO."""
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig
    from hermes_agentic_rl.yaml_config import build_trainer_config

    config = {
        "n_iters": 10,
        "group_size": 4,
        "clip_eps": 0.2,
    }
    cfg = build_trainer_config(config)
    assert isinstance(cfg, GRPOTrainerConfig)
    assert cfg.n_iters == 10


def test_gspo_yaml_file_loads():
    """Loading the gspo_echo.yaml config file should produce a valid config."""
    from pathlib import Path

    import yaml

    from hermes_agentic_rl.trainers.gspo_trainer import GSPOTrainerConfig

    config_path = Path(__file__).parent.parent / "configs" / "gspo_echo.yaml"
    if not config_path.exists():
        return  # skip if file not present

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["algo"] == "gspo"
    cfg = _build_trainer_config_typed(config, GSPOTrainerConfig)
    assert cfg.n_iters == 20
    assert cfg.log_ratio_clip == 40.0
