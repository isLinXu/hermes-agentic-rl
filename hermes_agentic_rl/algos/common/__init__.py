from hermes_agentic_rl.algos.common.advantage import (
    group_normalize_advantage,
    group_normalize_advantage_tensor,
)
from hermes_agentic_rl.algos.common.batch_prepare import (
    KLPenaltyResult,
    build_advantage_tensor,
    compute_entropy_bonus,
    compute_kl_penalty,
    mean_advantage_from_tensors,
    mean_reward_from_records,
    stack_old_logprobs,
)
from hermes_agentic_rl.algos.common.gae import (
    compute_gae,
    compute_gae_batched,
    terminal_token_rewards,
)
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs
from hermes_agentic_rl.algos.common.loss import (
    clipped_surrogate_loss,
    clipped_surrogate_loss_batched,
    clipped_value_loss,
    clipped_value_loss_batched,
)

__all__ = [
    "KLPenaltyResult",
    "build_advantage_tensor",
    "clipped_surrogate_loss",
    "clipped_surrogate_loss_batched",
    "clipped_value_loss",
    "clipped_value_loss_batched",
    "compute_entropy_bonus",
    "compute_gae",
    "compute_gae_batched",
    "compute_kl_penalty",
    "group_normalize_advantage",
    "group_normalize_advantage_tensor",
    "kl_from_logprobs",
    "mean_advantage_from_tensors",
    "mean_reward_from_records",
    "stack_old_logprobs",
    "terminal_token_rewards",
]
