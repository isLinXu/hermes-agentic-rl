from hermes_agentic_rl.rewards.aggregate import weighted_sum
from hermes_agentic_rl.rewards.base import BaseReward
from hermes_agentic_rl.rewards.composer import RewardComposer, RewardComposerConfig

__all__ = [
    "BaseReward",
    "RewardComposer",
    "RewardComposerConfig",
    "weighted_sum",
]

try:
    from hermes_agentic_rl.rewards.next_turn_feedback import (
        NextTurnFeedbackConfig,
        NextTurnFeedbackReward,
        score_feedback_messages,
    )

    __all__.extend(
        [
            "NextTurnFeedbackConfig",
            "NextTurnFeedbackReward",
            "score_feedback_messages",
        ]
    )
except Exception:
    pass

try:
    from hermes_agentic_rl.rewards.token_level_reward import (
        TokenRewardConfig,
        compute_combined_reward,
        compute_kl_penalty,
        compute_length_penalty,
        compute_outcome_reward,
        compute_token_advantages,
        compute_trajectory_token_rewards,
    )
    from hermes_agentic_rl.rewards.token_level_reward_component import TokenLevelReward

    __all__.extend(
        [
            "TokenLevelReward",
            "TokenRewardConfig",
            "compute_combined_reward",
            "compute_kl_penalty",
            "compute_length_penalty",
            "compute_outcome_reward",
            "compute_token_advantages",
            "compute_trajectory_token_rewards",
        ]
    )
except Exception:
    pass
