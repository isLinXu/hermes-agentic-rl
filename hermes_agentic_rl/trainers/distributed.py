"""FSDP / DDP distributed training support.

Provides:
  - :func:`wrap_for_distributed` — applies FSDP or DDP to a policy model
    based on configuration. Returns the wrapped model and a flag indicating
    whether the original parameters are still accessible.
  - :class:`DistributedConfig` — configuration dataclass.

Design:
  - FSDP (Fully Sharded Data Parallel): shards model params, gradients,
    and optimizer states across GPUs. Required for 7B+ models.
  - DDP (Distributed Data Parallel): replicates the model on each GPU,
    synchronizes gradients. Good for <1B models.
  - ``"none"`` mode: no wrapping (single-GPU or CPU training).
  - The wrapper is applied in the trainer's ``__init__``, BEFORE the
    optimizer is built, because FSDP manages its own parameter sharding.
  - ``sync_weights_to_vllm`` handles the special case of gathering FSDP
    shards into a single state_dict for vLLM rollout.

Optional dependencies: ``torch.distributed`` (built-in, but requires
``torchrun`` or similar launcher for multi-GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class DistributedConfig:
    """FSDP / DDP configuration.

    All fields have safe defaults for single-GPU training.
    """

    strategy: Literal["none", "ddp", "fsdp"] = "none"
    # ── FSDP options ──
    fsdp_sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"] = (
        "FULL_SHARD"
    )
    fsdp_cpu_offload: bool = False  # offload params to CPU (slower but saves VRAM)
    fsdp_backward_prefetch: Literal["BACKWARD_PRE", "BACKWARD_POST", "NO_PREFETCH"] = "BACKWARD_PRE"
    fsdp_use_orig_params: bool = True  # needed for optimizer state_dict checkpointing
    fsdp_auto_wrap_policy: str | None = None  # "transformer_layer" or None (manual)
    # ── DDP options ──
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25
    # ── common ──
    mixed_precision: Literal["fp16", "bf16", "fp32", "auto"] = "auto"
    # ── launch (set by torchrun automatically) ──
    local_rank: int = 0
    world_size: int = 1


def wrap_for_distributed(
    model: Any,  # nn.Module
    cfg: DistributedConfig,
) -> tuple[Any, bool]:
    """Apply FSDP or DDP wrapping to *model* in-place.

    Args:
        model: the policy model (nn.Module). If the backend uses a custom
            ``.model`` attribute (like HFCausalLMBackend), pass that.
        cfg: distributed config.

    Returns:
        (wrapped_model, is_wrapped) — ``is_wrapped`` is True when FSDP or
        DDP was applied. The caller should use ``wrapped_model`` for forward
        passes and parameter access.

    Raises:
        RuntimeError: if ``strategy != "none"`` but ``torch.distributed``
        is not initialized (caller forgot ``torchrun`` or ``mp.spawn``).
    """
    if cfg.strategy == "none":
        return model, False

    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            f"Distributed strategy '{cfg.strategy}' requires torch.distributed "
            f"to be initialized. Launch with: torchrun --nproc_per_node=N your_script.py"
        )

    if cfg.strategy == "ddp":
        wrapped = _apply_ddp(model, cfg)
        return wrapped, True

    if cfg.strategy == "fsdp":
        wrapped = _apply_fsdp(model, cfg)
        return wrapped, True

    raise ValueError(f"Unknown distributed strategy: {cfg.strategy!r}")


def _apply_ddp(model: Any, cfg: DistributedConfig) -> Any:
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [cfg.local_rank] if torch.cuda.is_available() else None
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=cfg.ddp_find_unused_parameters,
        bucket_cap_mb=cfg.ddp_bucket_cap_mb,
    )


def _apply_fsdp(model: Any, cfg: DistributedConfig) -> Any:
    import torch
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )

    # Sharding strategy.
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = strategy_map[cfg.fsdp_sharding_strategy]

    # Backward prefetch.
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        "NO_PREFETCH": None,
    }
    backward_prefetch = prefetch_map[cfg.fsdp_backward_prefetch]

    # CPU offload.
    cpu_offload = CPUOffload(offload_params=True) if cfg.fsdp_cpu_offload else None

    # Mixed precision for FSDP communication (not to be confused with AMP for
    # computation — FSDP can reduce grads in a lower dtype).
    from hermes_agentic_rl.trainers.mixed_precision import get_amp_dtype

    mp_dtype = get_amp_dtype(cfg.mixed_precision)
    mixed_precision = None
    if mp_dtype != torch.float32:
        mixed_precision = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )

    # Auto-wrap policy (optional).
    auto_wrap_policy = None
    if cfg.fsdp_auto_wrap_policy == "transformer_layer":
        # Try common Transformer layer class names.
        auto_wrap_policy = _transformer_auto_wrap_policy

    # Build FSDP wrapper.
    wrapped = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        mixed_precision=mixed_precision,
        use_orig_params=cfg.fsdp_use_orig_params,
        auto_wrap_policy=auto_wrap_policy,
        device_id=cfg.local_rank if torch.cuda.is_available() else None,
    )
    return wrapped


def _transformer_auto_wrap_policy(
    module: Any,
    recurse: bool,
    nonwrapped_numel: int,
) -> bool:
    """Auto-wrap policy that wraps Transformer decoder layers."""
    # Common layer class names across model architectures.
    layer_class_names = {
        "LlamaDecoderLayer",
        "Qwen2DecoderLayer",
        "GPT2Block",
        "GPTNeoXLayer",
        "MistralDecoderLayer",
        "GemmaDecoderLayer",
        "Phi3DecoderLayer",
        "BloomBlock",
        "OPTDecoderLayer",
        "FalconDecoderLayer",
        "MPTBlock",
    }
    class_name = module.__class__.__name__
    return class_name in layer_class_names


def gather_fsdp_state_dict(model: Any) -> dict[str, Any]:
    """Gather a full (un-sharded) state_dict from an FSDP-wrapped model.

    This is the slow but correct path — call it only for weight sync to
    vLLM (which runs once per N training iterations).

    Args:
        model: an FSDP-wrapped ``nn.Module``.

    Returns:
        Full state_dict with CPU tensors, suitable for ``model.load_state_dict()``.
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state = model.state_dict()
    return state
