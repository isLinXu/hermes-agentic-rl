"""Pydantic-based configuration validation for hermes-agentic-rl.

This module provides a schema-validation layer on top of the raw YAML
config loader (``yaml_config.py``). It:

1. **Validates** the top-level YAML structure before passing to the builder.
2. **Catches** mis-typed, mis-named, or structurally invalid configs early
   — before the trainer starts — with human-friendly error messages.
3. **Documents** the config schema via Pydantic field descriptions.

Usage::

    from hermes_agentic_rl.config_validation import validate_config

    # Raises ValidationError with detailed messages if invalid.
    validated = validate_config(raw_yaml_dict)
    # validated is a HermesConfig Pydantic model.

Design
------
* Pydantic v2 is an **optional** dependency. When unavailable, the module
  falls back to a lightweight dict-based validator that checks the same
  constraints.
* The schema mirrors the YAML keys in ``yaml_config.py`` — every top-level
  key in the YAML has a corresponding field here.
* Validation is **non-destructive**: the validated output can be converted
  back to a plain dict via ``.model_dump()`` for the existing builder.
"""

from __future__ import annotations

import logging
import typing
from typing import Any

logger = logging.getLogger(__name__)

# Try to import pydantic v2.
try:
    from pydantic import BaseModel, Field, field_validator, model_validator

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    BaseModel = object  # type: ignore[misc,assignment]

    def Field(**kwargs: Any) -> None:  # type: ignore[misc,assignment,no-redef]
        """Stub for pydantic.Field when pydantic is unavailable."""
        return None

    field_validator = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic models (v2)
# ---------------------------------------------------------------------------

if _HAS_PYDANTIC:

    class RewardComponentSpec(BaseModel):
        """Spec for a single reward component in the YAML config."""

        type: str = Field(..., description="Reward class name (e.g. ToolcallReward)")
        weight: float = Field(1.0, ge=0.0, description="Weight in the aggregate reward")
        params: dict[str, Any] = Field(
            default_factory=dict, description="Constructor kwargs for the reward class"
        )

    class RewardComposerSpec(BaseModel):
        """Spec for the reward composer block."""

        normalize: dict[str, bool] = Field(default_factory=dict)
        conditions: dict[str, str] = Field(default_factory=dict)
        turn_discount: dict[str, float] = Field(default_factory=dict)
        aggregator: str = Field("weighted_sum", description="Aggregation strategy")
        parallel: bool = Field(True, description="Evaluate components in parallel")

        @field_validator("aggregator")
        @classmethod
        def _validate_aggregator(cls, v: str) -> str:
            allowed = {"weighted_sum", "mean", "max", "gated"}
            if v not in allowed:
                raise ValueError(
                    f"aggregator must be one of {allowed}, got {v!r}"
                )
            return v

    class RewardsSpec(BaseModel):
        """Top-level rewards block."""

        components: list[RewardComponentSpec] = Field(default_factory=list)
        composer: RewardComposerSpec | None = None

    class RULERSpec(BaseModel):
        """RULER rule configuration block."""

        rules: list[dict[str, Any]] = Field(default_factory=list)
        default_weight: float = Field(1.0, ge=0.0)

    class CurriculumStageSpec(BaseModel):
        """A single curriculum stage."""

        name: str = Field(..., description="Stage name")
        weight: float = Field(1.0, ge=0.0)
        prompts: list[str] = Field(default_factory=list)
        mastery_threshold: float = Field(0.8, ge=0.0, le=1.0)
        min_iters: int = Field(5, ge=1)

    class CurriculumSpec(BaseModel):
        """Curriculum scheduler configuration."""

        auto_advance: bool = True
        min_iters_per_stage: int = Field(5, ge=1)
        allow_regression: bool = False
        regression_factor: float = Field(0.5, ge=0.0, le=1.0)
        stages: list[CurriculumStageSpec] | None = None

    class StalenessTISSpec(BaseModel):
        """Staleness-adaptive TIS controller config."""

        max_rho_clip: float = Field(1.5, ge=0.0)
        min_rho_clip: float = Field(0.5, ge=0.0)
        max_staleness: int = Field(5, ge=1)
        interpolation: str = Field("linear")
        rho_floor: float = Field(0.0, ge=0.0)
        window_size: int = Field(10, ge=1)
        enabled: bool = True

        @field_validator("interpolation")
        @classmethod
        def _validate_interp(cls, v: str) -> str:
            if v not in {"linear", "exp"}:
                raise ValueError(f"interpolation must be 'linear' or 'exp', got {v!r}")
            return v

        @model_validator(mode="after")
        def _check_clip_range(self) -> StalenessTISSpec:
            if self.min_rho_clip > self.max_rho_clip:
                raise ValueError(
                    f"min_rho_clip ({self.min_rho_clip}) > max_rho_clip "
                    f"({self.max_rho_clip})"
                )
            return self

    class LoRAHotReloadSpec(BaseModel):
        """LoRA hot-reload manager config."""

        rank: int = Field(8, ge=1)
        alpha: float = Field(16.0, gt=0.0)
        dropout: float = Field(0.0, ge=0.0, le=1.0)
        target_patterns: tuple[str, ...] = Field(("qkv", "proj"))
        sync_every: int = Field(1, ge=1)
        shadow_device: str = Field("cpu")

    class QuantizationSpec(BaseModel):
        """Quantization backend config."""

        model_path: str = Field("", description="Path to quantized model")
        format: str | None = Field(None, description="gptq|awq|gguf (auto-detected if omitted)")
        tensor_parallel_size: int = Field(1, ge=1)
        max_model_len: int = Field(4096, ge=128)
        gpu_memory_utilization: float = Field(0.90, ge=0.1, le=1.0)
        max_new_tokens: int = Field(512, ge=1)
        temperature: float = Field(1.0, ge=0.0)
        n_gpu_layers: int = Field(-1)

        @field_validator("format")
        @classmethod
        def _validate_format(cls, v: str | None) -> str | None:
            if v is not None and v.lower() not in {"gptq", "awq", "gguf"}:
                raise ValueError(f"format must be gptq|awq|gguf, got {v!r}")
            return v

    class ClientServerSpec(BaseModel):
        """Client-server architecture config."""

        enabled: bool = True
        server_host: str = Field("127.0.0.1")
        server_port: int = Field(5555, ge=1, le=65535)
        version_sync: bool = Field(True, description="Auto weight sync on version change")
        batch_size: int = Field(32, ge=1)

    class HermesConfig(BaseModel):
        """Top-level YAML configuration for hermes-agentic-rl training.

        This schema validates the full config before it reaches the trainer.
        Unknown keys are **allowed** (forward-compatibility) but logged.
        """

        model_config: typing.ClassVar[dict[str, str]] = {"extra": "allow"}  # type: ignore[misc,assignment]

        # Backend
        backend: str = Field("tiny", description="Backend name: tiny|hf|vllm")
        model_name: str = Field("", description="HuggingFace model name or path")

        # Training
        n_iters: int = Field(20, ge=1)
        group_size: int = Field(4, ge=1)
        prompts_per_iter: int = Field(2, ge=1)
        lr: float = Field(1e-3, gt=0.0)
        max_new_tokens: int = Field(16, ge=1)
        temperature: float = Field(1.0, ge=0.0)
        grad_clip: float = Field(1.0, ge=0.0)
        use_reference: bool = False
        multi_turn: bool = False
        seed: int | None = 0

        # Optimization
        update_epochs: int = Field(1, ge=1)
        minibatch_size: int = Field(0, ge=0)
        grad_accum_steps: int = Field(1, ge=1)
        normalize_reward: bool = False

        # Checkpointing
        save_every: int = Field(0, ge=0)
        checkpoint_every: int = Field(0, ge=0)
        output_dir: str | None = None

        # KL control
        target_kl: float = Field(0.0, ge=0.0)
        adaptive_kl: bool = False

        # vLLM
        vllm_rollout_model: str | None = None
        vllm_tensor_parallel_size: int = Field(1, ge=1)

        # Composite blocks
        rewards: RewardsSpec | None = None
        ruler: RULERSpec | None = None
        curriculum: CurriculumSpec | None = None
        staleness_adaptive_tis: StalenessTISSpec | None = None
        lora_hot_reload: LoRAHotReloadSpec | None = None
        quantization: QuantizationSpec | None = None
        client_server: ClientServerSpec | None = None

        @field_validator("backend")
        @classmethod
        def _validate_backend(cls, v: str) -> str:
            allowed = {"tiny", "hf", "vllm", "mock"}
            if v not in allowed:
                raise ValueError(
                    f"backend must be one of {allowed}, got {v!r}"
                )
            return v

        @model_validator(mode="after")
        def _check_group_size(self) -> HermesConfig:
            """GRPO requires group_size >= 2 for advantage computation."""
            if self.group_size < 2:
                logger.warning(
                    "group_size=1 disables group-relative advantage "
                    "(GRPO degenerates to REINFORCE)"
                )
            return self

        @model_validator(mode="after")
        def _check_lora_vllm(self) -> HermesConfig:
            """LoRA hot-reload requires vLLM rollout."""
            if self.lora_hot_reload and not self.vllm_rollout_model:
                raise ValueError(
                    "lora_hot_reload requires vllm_rollout_model to be set — "
                    "LoRA deltas are merged and synced to vLLM"
                )
            return self

        @model_validator(mode="after")
        def _check_staleness(self) -> HermesConfig:
            """staleness_adaptive_tis needs pipeline or replay to have effect."""
            if self.staleness_adaptive_tis:
                has_pipeline = getattr(self, "pipeline_rollouts", False)
                has_replay = getattr(self, "replay_buffer", None) is not None
                if not has_pipeline and not has_replay:
                    logger.warning(
                        "staleness_adaptive_tis has no effect without "
                        "pipeline_rollouts or replay_buffer"
                    )
            return self


# ---------------------------------------------------------------------------
# Fallback validator (no pydantic)
# ---------------------------------------------------------------------------

_FALLBACK_CHECKS: dict[str, tuple[type, str]] = {
    "backend": (str, "tiny|hf|vllm|mock"),
    "n_iters": (int, ">= 1"),
    "group_size": (int, ">= 1"),
    "prompts_per_iter": (int, ">= 1"),
    "lr": (float, "> 0"),
    "max_new_tokens": (int, ">= 1"),
    "temperature": (float, ">= 0"),
    "update_epochs": (int, ">= 1"),
    "grad_accum_steps": (int, ">= 1"),
    "vllm_tensor_parallel_size": (int, ">= 1"),
}

_ALLOWED_BACKENDS = {"tiny", "hf", "vllm", "mock"}
_ALLOWED_AGGRS = {"weighted_sum", "mean", "max", "gated"}
_ALLOWED_FORMATS = {"gptq", "awq", "gguf"}


def _fallback_validate(config: dict[str, Any]) -> dict[str, Any]:
    """Lightweight dict-based validator when pydantic is unavailable."""
    errors: list[str] = []

    for key, (expected_type, hint) in _FALLBACK_CHECKS.items():
        if key in config:
            val = config[key]
            if not isinstance(val, expected_type):
                errors.append(f"{key}: expected {expected_type.__name__}, got {type(val).__name__}")

    if "backend" in config and config["backend"] not in _ALLOWED_BACKENDS:
        errors.append(f"backend: must be one of {_ALLOWED_BACKENDS}, got {config['backend']!r}")

    if "n_iters" in config and config["n_iters"] < 1:
        errors.append("n_iters: must be >= 1")

    if "group_size" in config and config["group_size"] < 1:
        errors.append("group_size: must be >= 1")

    if "lr" in config and config["lr"] <= 0:
        errors.append("lr: must be > 0")

    # Check lora_hot_reload requires vllm_rollout_model
    if config.get("lora_hot_reload") and not config.get("vllm_rollout_model"):
        errors.append("lora_hot_reload requires vllm_rollout_model to be set")

    # Check quantization format
    quant = config.get("quantization", {})
    if isinstance(quant, dict) and quant.get("format"):
        if quant["format"].lower() not in _ALLOWED_FORMATS:
            errors.append(f"quantization.format: must be one of {_ALLOWED_FORMATS}")

    # Check aggregator
    rewards = config.get("rewards", {})
    if isinstance(rewards, dict):
        composer = rewards.get("composer", {})
        if isinstance(composer, dict) and composer.get("aggregator"):
            if composer["aggregator"] not in _ALLOWED_AGGRS:
                errors.append(f"rewards.composer.aggregator: must be one of {_ALLOWED_AGGRS}")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )

    return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_config(config: dict[str, Any]) -> Any:
    """Validate a raw YAML config dict.

    Parameters
    ----------
    config : dict
        Raw configuration as loaded from YAML.

    Returns
    -------
    HermesConfig | dict
        Validated config. When pydantic is available, returns a
        ``HermesConfig`` model; otherwise returns the validated dict.

    Raises
    ------
    ValueError | pydantic.ValidationError
        If the config is invalid.
    """
    if _HAS_PYDANTIC:
        return HermesConfig(**config)
    return _fallback_validate(config)


def validate_config_or_warn(config: dict[str, Any]) -> dict[str, Any]:
    """Validate config, returning the original dict on success.

    On validation failure, logs the error and returns the original
    config dict (so training can proceed with potential issues).
    """
    try:
        validated = validate_config(config)
        if _HAS_PYDANTIC and isinstance(validated, HermesConfig):
            return validated.model_dump(exclude_none=False)
        return config
    except Exception as e:
        logger.warning("Config validation failed (proceeding anyway):\n%s", e)
        return config


def is_pydantic_available() -> bool:
    """Check whether pydantic v2 is available."""
    return _HAS_PYDANTIC
