"""Pydantic schema for YAML / dict configs.

Optional dependency: install with ``pip install 'hermes-agentic-rl[config]'``.
When Pydantic is unavailable, :func:`hermes_agentic_rl.config.validate_config`
falls back to lightweight dict checks.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IntegrationName = Literal["fake", "hf", "vllm", "sglang", "openai", "anthropic"]
AlgoName = Literal[
    "grpo",
    "ppo",
    "hybrid",
    "opd",
    "rloo",
    "factored",
    "simpo",
    "gspo",
    "best_of_n",
]


class RuntimeConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    integration: IntegrationName = "fake"
    max_agent_turns: int = Field(default=20, ge=1)
    enabled_toolsets: list[str] = Field(default_factory=lambda: ["terminal", "file"])


class TrainerConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    algo: AlgoName | str = "grpo"
    n_iters: int | None = Field(default=None, ge=1)
    lr: float | None = Field(default=None, gt=0)
    group_size: int | None = Field(default=None, ge=1)

    @field_validator("algo", mode="before")
    @classmethod
    def _normalize_algo(cls, value: Any) -> Any:
        if value is None:
            return "grpo"
        return str(value).lower()


class RewardConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    aggregator: str = "weighted_sum"
    components: list[dict[str, Any]] = Field(default_factory=list)


class HermesRLConfigModel(BaseModel):
    """Top-level config envelope validated before training / eval CLIs run."""

    model_config = ConfigDict(extra="allow")

    runtime: RuntimeConfigModel = Field(default_factory=RuntimeConfigModel)
    environment: dict[str, Any] = Field(default_factory=dict)
    trainer: TrainerConfigModel = Field(default_factory=TrainerConfigModel)
    reward: RewardConfigModel = Field(default_factory=RewardConfigModel)
    backend: dict[str, Any] = Field(default_factory=dict)
    train_rl: dict[str, Any] = Field(default_factory=dict)


def validate_config_model(cfg: dict[str, Any]) -> HermesRLConfigModel:
    """Parse and validate *cfg* with Pydantic; raise on schema violations."""
    return HermesRLConfigModel.model_validate(cfg)


def config_model_to_dict(model: HermesRLConfigModel) -> dict[str, Any]:
    """Serialize a validated model back to a plain dict for legacy call sites."""
    return model.model_dump(mode="python")
