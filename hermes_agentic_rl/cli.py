"""YAML configuration loader for hermes-agentic-rl training.

Replaces the fragile argparse-based config in ``train_mtgrpo.py`` with a
declarative YAML file that maps directly to ``GRPOTrainerConfig`` fields.

Usage::

    # config.yaml
    backend: tiny
    model_name: Qwen/Qwen2.5-1.5B-Instruct
    n_iters: 20
    group_size: 4
    curriculum:
      auto_advance: true
      min_iters_per_stage: 10
    multi_turn: true
    multi_turn_credit:
      mode: discounted
      gamma: 0.9
    rewards:
      components:
        - type: ToolcallReward
          weight: 1.0
        - type: OutcomeReward
          weight: 1.0
      composer:
        normalize:
          toolcall_reward: true
          outcome_reward: true
        conditions:
          toolcall_reward: true  # always active
          outcome_reward: has_final_output  # prebuilt condition
        turn_discount:
          toolcall_reward: 0.95
    ruler:
      rules:
        - name: exact_match
          template: exact_match
          weight: 2.0
          params:
            gold_key: answer

    # Run:
    # python -m hermes_agentic_rl.cli --config config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig

logger = logging.getLogger(__name__)

# Try to import yaml; fall back to a minimal parser if unavailable.
try:
    import yaml
except ImportError:
    yaml = None
    logger.warning("PyYAML not installed — YAML config loading will fail")


# ---------------------------------------------------------------------------
# Reward component registry
# ---------------------------------------------------------------------------

_REWARD_REGISTRY: dict[str, type] = {}


def _register_reward(cls: type) -> type:
    """Decorator to register a reward component for YAML config."""
    _REWARD_REGISTRY[cls.__name__] = cls
    return cls


def _populate_registry() -> None:
    """Populate the reward registry with known components."""
    try:
        from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
        from hermes_agentic_rl.rewards.ruler import RULER
        from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
        _REWARD_REGISTRY["ToolcallReward"] = ToolcallReward
        _REWARD_REGISTRY["OutcomeReward"] = OutcomeReward
        _REWARD_REGISTRY["RULER"] = RULER
    except ImportError as e:
        logger.warning(f"Could not populate reward registry: {e}")


_populate_registry()


# ---------------------------------------------------------------------------
# Prebuilt condition functions
# ---------------------------------------------------------------------------

_PREBUILT_CONDITIONS: dict[str, Any] = {
    "true": lambda _item, _traj: True,
    "has_final_output": lambda _item, traj: bool(traj.final_output),
    "has_steps": lambda _item, traj: bool(traj.steps),
    "multi_turn": lambda _item, traj: traj.turns_used > 1,
}


def _resolve_condition(spec: Any) -> Any:
    """Resolve a condition spec to a callable.

    Supports:
    - String: look up in _PREBUILT_CONDITIONS
    - Boolean True: always-true condition
    - Boolean False: always-false condition (disables component)
    - Callable: return as-is
    - None: return None (always active, no gating)
    """
    if spec is None:
        return None
    if spec is True:
        return _PREBUILT_CONDITIONS["true"]
    if spec is False:
        return lambda _item, _traj: False
    if callable(spec):
        return spec
    if isinstance(spec, str) and spec in _PREBUILT_CONDITIONS:
        return _PREBUILT_CONDITIONS[spec]
    logger.warning(f"Unknown condition spec: {spec!r}; treating as always-true")
    return _PREBUILT_CONDITIONS["true"]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required for YAML config loading."
            " Install with: pip install pyyaml"
        )
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config)}")
    return config


def build_trainer_config(config: dict[str, Any]) -> GRPOTrainerConfig:
    """Build a ``GRPOTrainerConfig`` from a config dict.

    Maps top-level config keys to ``GRPOTrainerConfig`` fields.
    Unknown keys are ignored with a warning.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from YAML or programmatic).

    Returns
    -------
    GRPOTrainerConfig
    """
    # Get all valid field names from GRPOTrainerConfig
    from dataclasses import fields
    valid_fields = {f.name for f in fields(GRPOTrainerConfig)}

    kwargs: dict[str, Any] = {}
    unknown_keys: list[str] = []

    for key, value in config.items():
        # Skip non-config sections
        if key in ("backend", "model_name", "rewards", "ruler", "env"):
            continue
        if key in valid_fields:
            kwargs[key] = value
        else:
            unknown_keys.append(key)

    if unknown_keys:
        logger.warning(
            f"Unknown config keys ignored: {unknown_keys}. "
            f"Valid keys: {sorted(valid_fields)}"
        )

    # Convert output_dir string to Path
    if "output_dir" in kwargs and isinstance(kwargs["output_dir"], str):
        kwargs["output_dir"] = Path(kwargs["output_dir"])

    return GRPOTrainerConfig(**kwargs)


def build_reward_components(
    config: dict[str, Any],
) -> tuple[list[Any], dict[str, Any] | None]:
    """Build reward components and composer config from a config dict.

    Parameters
    ----------
    config : dict
        Configuration dictionary with a ``rewards`` section.

    Returns
    -------
    tuple
        (components_list, composer_config_dict_or_None)
    """
    rewards_cfg = config.get("rewards", {})
    if not rewards_cfg:
        return [], None

    components_spec = rewards_cfg.get("components", [])
    components: list[Any] = []

    for spec in components_spec:
        comp_type = spec.get("type", "")
        comp_weight = spec.get("weight", 1.0)
        comp_params = spec.get("params", {})

        if comp_type == "RULER":
            from hermes_agentic_rl.rewards.ruler import RULER
            ruler_cfg = config.get("ruler", {})
            if ruler_cfg:
                ruler_cfg.setdefault("rules", [])
                comp = RULER.from_config(ruler_cfg)
                comp.weight = comp_weight
            else:
                comp = RULER(weight=comp_weight)
        elif comp_type in _REWARD_REGISTRY:
            cls = _REWARD_REGISTRY[comp_type]
            comp = cls(weight=comp_weight, **comp_params)
        else:
            logger.warning(f"Unknown reward component type: {comp_type!r}; skipping")
            continue

        components.append(comp)

    # Build composer config
    composer_cfg = rewards_cfg.get("composer")
    if composer_cfg and isinstance(composer_cfg, dict):
        # Resolve condition strings to callables
        raw_conditions = composer_cfg.get("conditions", {})
        resolved_conditions = {
            name: _resolve_condition(spec)
            for name, spec in raw_conditions.items()
        }
        composer_config: dict[str, Any] | None = {
            "normalize": composer_cfg.get("normalize", {}),
            "conditions": resolved_conditions,
            "turn_discount": composer_cfg.get("turn_discount", {}),
            "aggregator": composer_cfg.get("aggregator", "weighted_sum"),
            "parallel": composer_cfg.get("parallel", True),
        }
    else:
        composer_config = None

    return components, composer_config


def build_backend(config: dict[str, Any]) -> Any:
    """Build a backend from config."""
    backend_type = config.get("backend", "tiny")
    if backend_type == "tiny":
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
        return TinyCausalLMBackend(
            TinyBackendConfig(
                seed=42,
                with_value_head=False,
                dim=32,
                n_heads=4,
                n_layers=2,
                max_len=256,
            )
        )
    elif backend_type == "hf":
        from hermes_agentic_rl.backends.hf_backend import HFBackend, HFBackendConfig
        return HFBackend(
            HFBackendConfig(
                model_name=config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                max_new_tokens=config.get("max_new_tokens", 256),
            )
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}")


def build_env(config: dict[str, Any]) -> Any:
    """Build a training environment from config."""
    from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset
    env_cfg = config.get("env", {})
    n_samples = env_cfg.get("n_samples", 100)
    seed = env_cfg.get("seed", 42)
    dataset = build_sim_tool_dataset(n=n_samples, seed=seed)
    return SimToolEnv(dataset=dataset)


def run_from_config(config: dict[str, Any]) -> None:
    """Run training from a config dict.

    This is the main entry point for YAML-driven training. It:

    1. Builds the backend, environment, and reward components.
    2. Creates a GRPOTrainerConfig (with curriculum if specified).
    3. Creates a GRPOTrainer and runs ``train()``.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    """
    from hermes_agentic_rl.rewards.composer import RewardComposer, RewardComposerConfig
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer

    # 1. Build components
    backend = build_backend(config)
    env = build_env(config)

    # 2. Build reward components + composer
    components, composer_cfg = build_reward_components(config)
    if components:
        if composer_cfg:
            reward_manager = RewardComposer(
                components=components,
                config=RewardComposerConfig(
                    normalize=composer_cfg.get("normalize", {}),
                    conditions=composer_cfg.get("conditions", {}),
                    turn_discount=composer_cfg.get("turn_discount", {}),
                    aggregator=composer_cfg.get("aggregator", "weighted_sum"),
                    parallel=composer_cfg.get("parallel", True),
                ),
            )
        else:
            from hermes_agentic_rl.core.reward_manager import RewardManager
            reward_manager = RewardManager(rewards=components)
    else:
        # Fallback: empty reward manager (shouldn't happen in practice)
        from hermes_agentic_rl.core.reward_manager import RewardManager
        reward_manager = RewardManager()
        logger.warning("No reward components configured — using empty RewardManager")

    # 3. Build trainer config
    trainer_cfg = build_trainer_config(config)

    # 4. Create and run trainer
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=trainer_cfg,
    )

    logger.info(f"Starting training: {trainer_cfg.n_iters} iters, "
                f"group_size={trainer_cfg.group_size}")

    trainer.train()
    logger.info("Training complete!")


def run_from_yaml(config_path: str | Path) -> None:
    """Load a YAML config and run training.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
    """
    config = load_config(config_path)
    run_from_config(config)
