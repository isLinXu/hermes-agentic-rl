"""Length penalty reward component.

This component is useful with sequence-level objectives such as GSPO or
Dr. GRPO-style aggregation, where you may want to decouple task reward from
response length bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward


@dataclass(slots=True)
class LengthPenaltyConfig:
    target_len: int = 512
    alpha: float = 0.1
    mode: Literal["linear", "quadratic"] = "linear"
    apply_on: Literal["response", "total"] = "response"


class LengthPenaltyReward(BaseReward):
    """Penalize responses longer than a target token length.

    Formula::

        penalty = -alpha * max(0, (length - target_len) / target_len)

    In ``quadratic`` mode the normalized excess is squared.
    """

    name = "length_penalty"

    def __init__(
        self,
        cfg: LengthPenaltyConfig | None = None,
        *,
        weight: float = 1.0,
    ) -> None:
        self.cfg = cfg or LengthPenaltyConfig()
        if self.cfg.mode not in {"linear", "quadratic"}:
            raise ValueError("LengthPenaltyConfig.mode must be 'linear' or 'quadratic'")
        if self.cfg.apply_on not in {"response", "total"}:
            raise ValueError("LengthPenaltyConfig.apply_on must be 'response' or 'total'")
        self.weight = float(weight)

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del item, tool_context
        response_len, prompt_len = _token_lengths(trajectory)
        length = response_len if self.cfg.apply_on == "response" else response_len + prompt_len
        target = max(1, int(self.cfg.target_len))
        excess = max(0.0, (float(length) - float(target)) / float(target))
        if self.cfg.mode == "quadratic":
            excess = excess * excess
        score = -float(self.cfg.alpha) * excess
        return RewardResult(
            name=self.name,
            score=score,
            reason=f"{self.cfg.apply_on}_tokens={length} target={target}",
            weight=self.weight,
            metadata={
                "length": int(length),
                "response_tokens": int(response_len),
                "prompt_tokens": int(prompt_len),
                "target_len": int(target),
                "alpha": float(self.cfg.alpha),
                "mode": self.cfg.mode,
                "apply_on": self.cfg.apply_on,
                "excess_ratio": float(excess),
            },
        )


def _token_lengths(trajectory: Trajectory) -> tuple[int, int]:
    runtime = trajectory.metadata.get("runtime")
    rl_meta = runtime.get("rl") if isinstance(runtime, dict) else None
    if isinstance(rl_meta, dict):
        response_ids = rl_meta.get("response_ids")
        prompt_ids = rl_meta.get("prompt_ids")
        response_len = len(response_ids) if isinstance(response_ids, list) else 0
        prompt_len = len(prompt_ids) if isinstance(prompt_ids, list) else 0
        return response_len, prompt_len

    text = trajectory.final_output or ""
    return len(text.split()) if text else 0, 0
