"""Offline algorithms: BC (Behavior Cloning), DPO (Direct Preference Optimization).

These trainers consume JSONL samples (see ``replay_buffer.ReplayBuffer``)
rather than running rollouts. They share the same ``LLMBackend`` protocol,
so a model pre-trained with BC / DPO is a drop-in replacement for the initial
policy passed to GRPOTrainer / PPOTrainer.
"""

from hermes_agentic_rl.offline.bc import BCConfig, BCTrainer
from hermes_agentic_rl.offline.dpo import DPOConfig, DPOTrainer
from hermes_agentic_rl.offline.replay_buffer import (
    DPOPair,
    ReplayBuffer,
    TrainSample,
)

__all__ = [
    "BCConfig",
    "BCTrainer",
    "DPOConfig",
    "DPOPair",
    "DPOTrainer",
    "ReplayBuffer",
    "TrainSample",
]
