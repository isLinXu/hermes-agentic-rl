"""RL algorithms: PPO / GRPO / DPO (MVP starts with GRPO, PPO in v0.3).

Each algorithm operates on a `RolloutBatch` (a list of `RolloutRecord`s that
carry prompt_ids, response_ids, old_logprobs, and a scalar reward) and returns
`AlgoUpdateStats` (loss, kl, entropy, etc.). The algorithm does NOT own the
optimizer; the Trainer does. Algorithms are pure torch functions so they are
easy to test.
"""

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.ppo import PPO, PPOConfig

__all__ = [
    "PPO",
    "GRPO",
    "PPOConfig",
    "GRPOConfig",
    "AlgoUpdateStats",
    "BaseAlgo",
    "RolloutBatch",
    "RolloutRecord",
]
