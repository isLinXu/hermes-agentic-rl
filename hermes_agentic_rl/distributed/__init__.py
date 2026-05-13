"""Distributed rollout primitives.

`MPRolloutPool`: multiprocessing-based rollout actor pool. Learner process
broadcasts policy weights → N worker processes run rollouts in parallel →
records are returned over a queue. Zero external deps (no Ray required).
"""

from hermes_agentic_rl.distributed.mp_pool import (
    MPRolloutPool,
    MPRolloutPoolConfig,
    RolloutTask,
)

__all__ = ["MPRolloutPool", "MPRolloutPoolConfig", "RolloutTask"]
