"""Real RL trainers (gradient-based).

These trainers own a policy (nn.Module), an optimizer, and perform actual
parameter updates. For rollout → JSONL serialization (Atropos-style), see
`hermes_agentic_rl.exporters` instead.
"""

# Back-compat re-export (deprecated).
from hermes_agentic_rl.exporters.atropos_jsonl import AtroposGrpoTrainer
from hermes_agentic_rl.trainers.base import BaseTrainer

# Real trainers (optional torch dep). Import lazily so CPU-only users without
# torch can still use the exporter path.
try:
    from hermes_agentic_rl.trainers.grpo_trainer import (
        GRPOTrainer,
        GRPOTrainerConfig,
        GRPOTrainStats,
    )
    from hermes_agentic_rl.trainers.on_policy import (
        OnPolicyTrainer,
        TrainStats,
    )
    from hermes_agentic_rl.trainers.on_policy_config import OnPolicyTrainerConfig
    from hermes_agentic_rl.trainers.ppo_trainer import (
        PPOTrainer,
        PPOTrainerConfig,
        PPOTrainStats,
    )

    _HAS_RL = True
except Exception:  # pragma: no cover - torch missing fallback
    _HAS_RL = False

__all__ = ["AtroposGrpoTrainer", "BaseTrainer"]
if _HAS_RL:
    __all__.extend(
        [
            "GRPOTrainStats",
            "GRPOTrainer",
            "GRPOTrainerConfig",
            "OnPolicyTrainer",
            "OnPolicyTrainerConfig",
            "PPOTrainStats",
            "PPOTrainer",
            "PPOTrainerConfig",
            "TrainStats",
        ]
    )
