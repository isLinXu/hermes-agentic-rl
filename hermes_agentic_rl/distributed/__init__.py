"""Distributed rollout primitives.

`MPRolloutPool`: multiprocessing-based rollout actor pool. Learner process
broadcasts policy weights → N worker processes run rollouts in parallel →
records are returned over a queue. Zero external deps (no Ray required).

`FaultTolerantRolloutPool`: wraps MPRolloutPool with retry, restart, and
elastic scaling for long-running RL training jobs.

`ModelParallelConfig` / `apply_model_parallel`: training-side tensor and
pipeline parallelism strategy interface.
"""

from hermes_agentic_rl.distributed.fault_tolerant_pool import (
    ElasticScalingConfig,
    FaultTolerantPoolConfig,
    FaultTolerantRolloutPool,
)
from hermes_agentic_rl.distributed.model_parallel import (
    HybridParallelStrategy,
    ModelParallelConfig,
    ModelParallelStrategy,
    PipelineParallelStrategy,
    TensorParallelStrategy,
    apply_model_parallel,
    compute_parallel_config,
)
from hermes_agentic_rl.distributed.mp_pool import (
    MPRolloutPool,
    MPRolloutPoolConfig,
    RolloutTask,
)

__all__ = [
    "ElasticScalingConfig",
    "FaultTolerantPoolConfig",
    "FaultTolerantRolloutPool",
    "HybridParallelStrategy",
    "MPRolloutPool",
    "MPRolloutPoolConfig",
    "ModelParallelConfig",
    "ModelParallelStrategy",
    "PipelineParallelStrategy",
    "RolloutTask",
    "TensorParallelStrategy",
    "apply_model_parallel",
    "compute_parallel_config",
]
