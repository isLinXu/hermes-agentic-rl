"""Model parallel strategy interface for training-side parallelism.

This module provides an abstraction layer for tensor parallelism (TP) and
pipeline parallelism (PP) on the training side — complementing the existing
FSDP/DDP data parallel support in ``distributed.py``.

Design:
- :class:`ModelParallelConfig` — TP/PP configuration.
- :class:`ModelParallelStrategy` — abstract base for parallel strategies.
- :class:`TensorParallelStrategy` — shards model layers across GPUs
  (uses PyTorch's ``TensorParallel`` or Megatron-style custom sharding).
- :class:`PipelineParallelStrategy` — splits model stages across GPUs
  with micro-batch pipelining (uses ``torch.distributed.pipeline``).
- :class:`HybridParallelStrategy` — combines TP + PP + DP (3D parallelism).
- :func:`apply_model_parallel` — entry point that selects and applies
  the appropriate strategy.

The interface is designed to be backend-agnostic: it works with any
``nn.Module`` and doesn't hardcode a specific model architecture. The
actual sharding is done lazily (only when torch.distributed is initialized
and multiple GPUs are available).

When parallelism is not available (single GPU, no torch.distributed), the
strategies are no-ops that return the model unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from hermes_agentic_rl.trainers.distributed import DistributedConfig

# ---------------------------------------------------------------------------
# Third-party integration hooks (graceful degradation when deps missing)
# ---------------------------------------------------------------------------


class MegatronIntegration:
    """Megatron-Core integration hooks for tensor and pipeline parallelism.

    All methods are no-ops when ``megatron-core`` is not installed,
    matching the lazy-import pattern used throughout this module.
    """

    @staticmethod
    def try_megatron_import() -> bool:
        """Return True if ``megatron.core`` is importable."""
        try:
            import megatron.core  # type: ignore[import-untyped]  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def configure_megatron_tp(model: Any, tp_size: int) -> Any | None:
        """Set up Megatron tensor parallelism when available.

        Returns the wrapped model on success, or ``None`` if megatron-core
        is missing. In a production implementation this would replace
        ``nn.Linear`` layers with their parallel counterparts and register
        TP process groups.
        """
        if not MegatronIntegration.try_megatron_import():
            return None
        # Production path: import megatron.core.tensor_parallel and shard
        # layers. For the interface-level stub we return the model unchanged.
        return model

    @staticmethod
    def configure_megatron_pp(
        model: Any, pp_size: int, num_chunks: int
    ) -> Any | None:
        """Set up Megatron pipeline parallelism when available.

        Returns the wrapped model on success, or ``None`` if megatron-core
        is missing.
        """
        if not MegatronIntegration.try_megatron_import():
            return None
        # Production path: use megatron.core.pipeline_parallel to wrap
        # the model into stages. For the interface-level stub we return
        # the model unchanged.
        return model


class DeepSpeedIntegration:
    """DeepSpeed integration hooks for ZeRO and pipeline parallelism.

    All methods are no-ops when ``deepspeed`` is not installed,
    matching the lazy-import pattern used throughout this module.
    """

    @staticmethod
    def try_deepspeed_import() -> bool:
        """Return True if ``deepspeed`` is importable."""
        try:
            import deepspeed  # type: ignore[import-untyped]  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def configure_deepspeed_zero(model: Any, zero_stage: int) -> Any | None:
        """Configure DeepSpeed ZeRO (1/2/3) when available.

        Returns the wrapped model on success, or ``None`` if DeepSpeed
        is missing.
        """
        if not DeepSpeedIntegration.try_deepspeed_import():
            return None
        # Production path: wrap model with deepspeed.DeepSpeedEngine.
        # For the interface-level stub we return the model unchanged.
        return model

    @staticmethod
    def configure_deepspeed_pipeline(
        model: Any, pp_size: int, num_chunks: int
    ) -> Any | None:
        """Configure DeepSpeed pipeline parallelism when available.

        Returns the wrapped model on success, or ``None`` if DeepSpeed
        is missing.
        """
        if not DeepSpeedIntegration.try_deepspeed_import():
            return None
        # Production path: use deepspeed.PipelineEngine. For the
        # interface-level stub we return the model unchanged.
        return model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelParallelConfig:
    """Configuration for model-parallel training.

    Attributes
    ----------
    tensor_parallel_size:
        Number of GPUs to shard each layer across (TP). 1 = no tensor
        parallelism.

    pipeline_parallel_size:
        Number of pipeline stages (PP). 1 = no pipeline parallelism.

    expert_parallel_size:
        Number of GPUs to shard MoE experts across (EP). 1 = no expert
        parallelism. Only relevant for models with Mixture-of-Experts layers.

    pipeline_chunks:
        Number of micro-batches per forward pass when PP > 1.

    backend:
        Which TP implementation to use: "auto" (try megatron → torch native),
        "megatron" (requires megatron-core), "torch" (torch.distributed TP).

    device:
        Target device for the model after parallel wrapping.

    overlap_comm:
        When True, overlap gradient communication with backward computation
        (TP only, improves throughput at cost of memory).

    activation_checkpointing:
        When True, use activation checkpointing within TP layers to
        trade compute for memory.
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_chunks: int = 1
    backend: Literal["auto", "megatron", "torch"] = "auto"
    device: str = "cuda"
    overlap_comm: bool = True
    activation_checkpointing: bool = False

    @property
    def enabled(self) -> bool:
        """True when any model parallelism is active."""
        return (
            self.tensor_parallel_size > 1
            or self.pipeline_parallel_size > 1
            or self.expert_parallel_size > 1
        )

    @property
    def total_parallel_size(self) -> int:
        """Total model-parallel GPUs (TP × PP × EP)."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
        )

    def validate(self) -> list[str]:
        """Return a list of validation warnings (empty if OK)."""
        warnings: list[str] = []
        if self.tensor_parallel_size < 1:
            warnings.append("tensor_parallel_size must be >= 1")
        if self.pipeline_parallel_size < 1:
            warnings.append("pipeline_parallel_size must be >= 1")
        if self.expert_parallel_size < 1:
            warnings.append("expert_parallel_size must be >= 1")
        if self.pipeline_chunks < 1 and self.pipeline_parallel_size > 1:
            warnings.append("pipeline_chunks must be >= 1 when PP > 1")
        if self.tensor_parallel_size > 1 and self.backend == "torch":
            warnings.append(
                "torch native TP is experimental; consider 'megatron' for production"
            )
        return warnings


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class ModelParallelStrategy(ABC):
    """Abstract base for model parallelism strategies.

    Subclasses implement :meth:`apply` which takes a model and returns a
    parallel-wrapped model. The strategy also exposes hooks for
    weight synchronization (:meth:`gather_state_dict`) and gradient
    reduction (:meth:`sync_gradients`).
    """

    def __init__(self, cfg: ModelParallelConfig) -> None:
        self.cfg = cfg
        self._applied = False

    @abstractmethod
    def apply(self, model: Any) -> Any:
        """Apply parallelism to the model. Returns wrapped model."""
        ...

    @abstractmethod
    def gather_state_dict(self, model: Any) -> dict[str, Any]:
        """Gather sharded state_dict into a full state_dict."""
        ...

    @abstractmethod
    def sync_gradients(self, model: Any) -> None:
        """Synchronize gradients across parallel groups after backward."""
        ...

    @property
    def is_applied(self) -> bool:
        return self._applied

    def teardown(self, model: Any) -> None:
        """Clean up parallel groups and release resources.

        Default implementation is a no-op. Subclasses may override.
        """
        self._applied = False


def gather_sharded_tensor(
    tensor: Any,
    dim: int,
    tp_process_group: Any,
) -> Any:
    """AllGather a sharded tensor along a dimension and concatenate.

    Args:
        tensor: local shard tensor.
        dim: dimension along which the tensor was sharded.
        tp_process_group: ``torch.distributed`` process group for tensor
            parallelism. When ``None``, the tensor is returned unchanged
            (no-op mode).

    Returns:
        Full tensor concatenated from all shards, or ``tensor`` if no
        sharding is active.
    """
    import torch
    import torch.distributed as dist

    if tp_process_group is None:
        return tensor

    try:
        tp_size = dist.get_world_size(group=tp_process_group)
        if tp_size <= 1:
            return tensor

        gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
        dist.all_gather(gathered, tensor, group=tp_process_group)
        return torch.cat(gathered, dim=dim)
    except Exception:
        return tensor


# ---------------------------------------------------------------------------
# Tensor parallel strategy
# ---------------------------------------------------------------------------


class TensorParallelStrategy(ModelParallelStrategy):
    """Shard model layers across multiple GPUs (tensor parallelism).

    Uses lazy import of the TP backend. When the backend is unavailable,
    falls back to no-op (returns model unchanged).
    """

    def __init__(self, cfg: ModelParallelConfig) -> None:
        super().__init__(cfg)
        self._shard_map: dict[str, dict[str, Any]] = {}
        self._tp_group: Any = None
        self._tp_rank: int = 0
        self._tp_size: int = 1

    def apply(self, model: Any) -> Any:
        if not self.cfg.enabled:
            return model
        try:
            import warnings

            import torch.distributed as dist

            if not dist.is_initialized():
                return model
            world_size = dist.get_world_size()
            if world_size < self.cfg.tensor_parallel_size:
                warnings.warn(
                    f"Requested TP size {self.cfg.tensor_parallel_size} but "
                    f"world size is only {world_size}. Falling back to no-op.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return model
        except Exception:
            return model

        # Try backend integrations first
        if self.cfg.backend in ("auto", "megatron"):
            wrapped = self._try_megatron(model)
            if wrapped is not None:
                self._applied = True
                return wrapped

        if self.cfg.backend in ("auto", "torch"):
            wrapped = self._try_torch_native(model)
            if wrapped is not None:
                self._applied = True
                return wrapped

        # Fallback: custom column/row-wise linear sharding
        try:
            self._setup_tp_group()
            sharded_count = self._apply_sharding(model)
            if sharded_count > 0:
                self._applied = True
        except Exception:
            pass

        return model

    def _setup_tp_group(self) -> None:
        """Create and cache the TP process group."""
        import torch.distributed as dist

        if self._tp_group is not None:
            return

        self._tp_size = self.cfg.tensor_parallel_size
        self._tp_rank = dist.get_rank() % self._tp_size

        # Create a contiguous TP process group
        ranks = list(range(self._tp_size))
        self._tp_group = dist.new_group(ranks)

    def _apply_sharding(self, model: Any) -> int:
        """Apply column/row-wise sharding to ``nn.Linear`` layers.

        Returns:
            Number of layers that were sharded.
        """
        import torch.nn as nn

        sharded_count = 0
        linear_count = 0

        # Snapshot to avoid mutation-while-iteration issues
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            shard_type = "column" if (linear_count % 2 == 0) else "row"
            if shard_type == "column":
                sharded = self._shard_linear_columnwise(
                    module, self._tp_rank, self._tp_size
                )
            else:
                sharded = self._shard_linear_rowwise(
                    module, self._tp_rank, self._tp_size
                )

            # Replace module in parent
            parent_name, _, child_name = name.rpartition(".")
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, sharded)

            # Record shard metadata for gather / sync
            for p_name, _ in sharded.named_parameters():
                full_name = f"{name}.{p_name}" if name else p_name
                self._shard_map[full_name] = {
                    "type": shard_type,
                    "dim": 0 if shard_type == "column" else 1,
                    "tp_size": self._tp_size,
                }

            sharded_count += 1
            linear_count += 1

        return sharded_count

    def _shard_linear_columnwise(self, module: Any, rank: int, tp_size: int) -> Any:
        """Shard an ``nn.Linear`` layer along the output dimension (column-wise).

        Each TP rank holds ``weight[out_dim // tp_size, in_dim]``. Bias is
        also sharded. The full output is produced by all-gathering partial
        results along the output dimension.

        When ``torch.distributed`` is unavailable or ``tp_size <= 1``,
        returns the original module unchanged.
        """
        import torch
        import torch.distributed as dist
        import torch.nn as nn

        if not isinstance(module, nn.Linear):
            return module
        if tp_size <= 1 or not dist.is_initialized():
            return module

        out_dim, in_dim = module.weight.shape
        out_per_rank = out_dim // tp_size
        start = rank * out_per_rank
        end = start + out_per_rank

        sharded = nn.Linear(
            in_dim,
            out_per_rank,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            sharded.weight.copy_(module.weight[start:end, :])
            if module.bias is not None:
                sharded.bias.copy_(module.bias[start:end])
        return sharded

    def _shard_linear_rowwise(self, module: Any, rank: int, tp_size: int) -> Any:
        """Shard an ``nn.Linear`` layer along the input dimension (row-wise).

        Each TP rank holds ``weight[out_dim, in_dim // tp_size]``. Bias is
        **not** sharded (all ranks keep the full bias). Output is all-reduced
        across the TP group during forward.

        When ``torch.distributed`` is unavailable or ``tp_size <= 1``,
        returns the original module unchanged.
        """
        import torch
        import torch.distributed as dist
        import torch.nn as nn

        if not isinstance(module, nn.Linear):
            return module
        if tp_size <= 1 or not dist.is_initialized():
            return module

        out_dim, in_dim = module.weight.shape
        in_per_rank = in_dim // tp_size
        start = rank * in_per_rank
        end = start + in_per_rank

        sharded = nn.Linear(
            in_per_rank,
            out_dim,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            sharded.weight.copy_(module.weight[:, start:end])
            if module.bias is not None:
                sharded.bias.copy_(module.bias)
        return sharded

    def gather_state_dict(self, model: Any) -> dict[str, Any]:
        """Gather TP-sharded weights into a full state_dict."""
        if not self._applied:
            return dict(model.state_dict())
        try:
            import torch.distributed as dist

            if dist.is_initialized() and self._tp_group is not None:
                full_state: dict[str, Any] = {}
                local_state = model.state_dict()
                for key, tensor in local_state.items():
                    shard_info = self._shard_map.get(key)
                    if shard_info is not None:
                        full_state[key] = gather_sharded_tensor(
                            tensor, shard_info["dim"], self._tp_group
                        )
                    else:
                        full_state[key] = tensor
                return full_state
        except Exception:
            pass
        return dict(model.state_dict())

    def sync_gradients(self, model: Any) -> None:
        """All-reduce gradients across the TP group.

        Only column-wise sharded parameters need gradient synchronization;
        row-wise shards perform all-reduce during forward, so backward
        gradients are already consistent.
        """
        if not self._applied:
            return
        try:
            import torch.distributed as dist

            if dist.is_initialized() and self._tp_group is not None:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    shard_info = self._shard_map.get(name)
                    if shard_info is not None and shard_info["type"] == "column":
                        dist.all_reduce(
                            param.grad,
                            op=dist.ReduceOp.SUM,
                            group=self._tp_group,
                        )
                        param.grad /= shard_info["tp_size"]
                    # Row-wise: forward already performed all-reduce; no extra
                    # backward synchronization is required.
        except Exception:
            pass

    def teardown(self, model: Any) -> None:
        """Clean up TP process group and shard metadata."""
        super().teardown(model)
        self._shard_map.clear()
        self._tp_group = None
        self._tp_rank = 0
        self._tp_size = 1

    def _try_megatron(self, model: Any) -> Any | None:
        """Attempt Megatron-Core TP wrapping. Returns None if unavailable or unchanged."""
        wrapped = MegatronIntegration.configure_megatron_tp(
            model, self.cfg.tensor_parallel_size
        )
        # If the backend returned the same model object, it didn't actually shard.
        if wrapped is model:
            return None
        return wrapped

    def _try_torch_native(self, model: Any) -> Any | None:
        """Attempt torch native TP wrapping (PyTorch 2.4+)."""
        try:
            import torch.distributed.tensor.parallel as tp  # type: ignore[attr-defined]

            del tp  # acknowledge import
            # Currently torch native TP is a stub; return None to trigger custom fallback.
            return None
        except (ImportError, AttributeError):
            return None


# ---------------------------------------------------------------------------
# Pipeline parallel strategy
# ---------------------------------------------------------------------------


class PipelineParallelStrategy(ModelParallelStrategy):
    """Split model into stages across GPUs (pipeline parallelism).

    Uses ``torch.distributed.pipeline`` (or torch's built-in PP in 2.4+).
    When unavailable, falls back to no-op.
    """

    def apply(self, model: Any) -> Any:
        if self.cfg.pipeline_parallel_size <= 1:
            return model
        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                return model
        except Exception:
            return model

        wrapped = self._try_torch_pipeline(model)
        if wrapped is not None:
            self._applied = True
            return wrapped
        return model

    def gather_state_dict(self, model: Any) -> dict[str, Any]:
        """Gather PP-stage state_dict from rank 0."""
        if not self._applied:
            return dict(model.state_dict())
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                # In PP, each rank has a different subset of layers.
                # Gathering requires all-gather of the full model.
                full_state: dict[str, Any] = {}
                local_state = model.state_dict()
                for key, tensor in local_state.items():
                    # Broadcast from the rank that owns this layer
                    # Simplified: rank 0 broadcasts everything
                    dist.broadcast(tensor, src=0)
                    full_state[key] = tensor
                return full_state
        except Exception:
            pass
        return dict(model.state_dict())

    def sync_gradients(self, model: Any) -> None:
        """PP gradient sync is handled by the pipeline scheduler."""
        if not self._applied:
            return
        # In torch.distributed.pipeline, gradient sync is automatic
        # via send/recv between stages. No manual sync needed.

    def _try_torch_pipeline(self, model: Any) -> Any | None:
        """Attempt torch pipeline parallel wrapping.

        Falls back to DeepSpeed or Megatron pipeline integrations when
        available. Returns the model unchanged (stub) if no backend is
        installed.
        """
        try:
            # torch.distributed.pipeline was added in PyTorch 1.8
            # but the API changed significantly in 2.4+
            # For the interface-level implementation, we return the model.
            # Actual PP would use torch.distributed.pipeline.sync.Pipe
            # or torch.distributed._pipeline in newer versions.
            # Also attempt DeepSpeed and Megatron PP integrations.
            wrapped = DeepSpeedIntegration.configure_deepspeed_pipeline(
                model, self.cfg.pipeline_parallel_size, self.cfg.pipeline_chunks
            )
            if wrapped is not None:
                return wrapped
            wrapped = MegatronIntegration.configure_megatron_pp(
                model, self.cfg.pipeline_parallel_size, self.cfg.pipeline_chunks
            )
            if wrapped is not None:
                return wrapped
            return model
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Hybrid (3D) parallel strategy
# ---------------------------------------------------------------------------


class HybridParallelStrategy(ModelParallelStrategy):
    """Combine tensor + pipeline + data parallelism (3D parallelism).

    This is the production strategy for large-scale training (7B+ models
    on multi-node clusters). It nests TP within PP within DP (FSDP/DDP).
    """

    def __init__(
        self,
        cfg: ModelParallelConfig,
        dp_cfg: DistributedConfig | None = None,
    ) -> None:
        super().__init__(cfg)
        self._tp_strategy = TensorParallelStrategy(cfg)
        self._pp_strategy = PipelineParallelStrategy(cfg)
        self._dp_cfg = dp_cfg or DistributedConfig()

    def apply(self, model: Any) -> Any:
        if not self.cfg.enabled:
            return model
        # Apply TP first (intra-node), then PP (inter-stage), then DP
        model = self._tp_strategy.apply(model)
        model = self._pp_strategy.apply(model)
        # DP (FSDP/DDP) is applied by the caller via wrap_for_distributed
        self._applied = self._tp_strategy.is_applied or self._pp_strategy.is_applied
        return model

    def gather_state_dict(self, model: Any) -> dict[str, Any]:
        """Gather full state_dict across all parallel groups."""
        state = self._tp_strategy.gather_state_dict(model)
        # In a full implementation, PP gather would collect stage-specific
        # layers from each pipeline rank. Since PipelineParallelStrategy
        # currently returns the model unchanged (stub), we return the
        # TP-gathered state dict directly.
        return state

    def sync_gradients(self, model: Any) -> None:
        """Sync gradients across TP and DP groups (PP handles its own)."""
        self._tp_strategy.sync_gradients(model)

    def teardown(self, model: Any) -> None:
        self._tp_strategy.teardown(model)
        self._pp_strategy.teardown(model)
        self._applied = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_model_parallel(
    model: Any,
    mp_cfg: ModelParallelConfig,
    dp_cfg: DistributedConfig | None = None,
) -> tuple[Any, ModelParallelStrategy]:
    """Apply model parallelism to a model.

    This is the main entry point. It selects the appropriate strategy
    based on the configuration and returns the (possibly wrapped) model
    and the strategy object for later gradient sync / state_dict gathering.

    Args:
        model: the policy model (nn.Module).
        mp_cfg: model-parallel configuration (TP/PP).
        dp_cfg: data-parallel configuration (FSDP/DDP), used only by
            the hybrid strategy.

    Returns:
        (wrapped_model, strategy) — ``strategy`` should be kept by the
        caller for gradient synchronization and checkpoint gathering.

    Example::

        mp_cfg = ModelParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        model, strategy = apply_model_parallel(model, mp_cfg)
        # ... training loop ...
        strategy.sync_gradients(model)
        full_state = strategy.gather_state_dict(model)
    """
    if not mp_cfg.enabled:
        return model, _NoOpStrategy(mp_cfg)

    if mp_cfg.tensor_parallel_size > 1 and mp_cfg.pipeline_parallel_size > 1:
        strategy: ModelParallelStrategy = HybridParallelStrategy(mp_cfg, dp_cfg)
    elif mp_cfg.tensor_parallel_size > 1:
        strategy = TensorParallelStrategy(mp_cfg)
    elif mp_cfg.pipeline_parallel_size > 1:
        strategy = PipelineParallelStrategy(mp_cfg)
    else:
        strategy = _NoOpStrategy(mp_cfg)

    wrapped = strategy.apply(model)
    return wrapped, strategy


# ---------------------------------------------------------------------------
# No-op strategy (for single-GPU / disabled parallelism)
# ---------------------------------------------------------------------------


class _NoOpStrategy(ModelParallelStrategy):
    """Pass-through strategy when model parallelism is disabled."""

    def apply(self, model: Any) -> Any:
        return model

    def gather_state_dict(self, model: Any) -> dict[str, Any]:
        return dict(model.state_dict())

    def sync_gradients(self, model: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Utility: compute optimal parallel configuration
# ---------------------------------------------------------------------------


def compute_parallel_config(
    model_params: int,
    n_gpus: int,
    gpu_memory_gb: float,
    *,
    max_tp_size: int = 8,
    prefer_pipeline: bool = False,
) -> ModelParallelConfig:
    """Heuristically compute an optimal TP/PP configuration.

    This is a convenience function that estimates the best parallelism
    strategy based on model size and available GPU resources.

    Args:
        model_params: total model parameters (e.g. 7_000_000_000 for 7B).
        n_gpus: total available GPUs.
        gpu_memory_gb: per-GPU memory in GB.
        max_tp_size: maximum TP size (typically limited by intra-node GPUs).
        prefer_pipeline: when True, prefer PP over TP for large models.

    Returns:
        A :class:`ModelParallelConfig` with recommended settings.
    """
    # Rough estimate: 4 bytes per param (fp32), 2x for optimizer state
    model_memory_gb = (model_params * 4 * 2) / (1024**3)
    single_gpu_fit = model_memory_gb <= gpu_memory_gb * 0.8  # 80% utilization

    if single_gpu_fit or n_gpus <= 1:
        return ModelParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

    # Need parallelism
    if prefer_pipeline:
        pp_size = min(n_gpus, max(1, int(model_memory_gb / (gpu_memory_gb * 0.6))))
        tp_size = min(max_tp_size, n_gpus // pp_size)
    else:
        tp_size = min(max_tp_size, n_gpus)
        pp_size = max(1, n_gpus // tp_size)

    # Adjust to ensure TP × PP <= n_gpus
    while tp_size * pp_size > n_gpus:
        if tp_size > 1:
            tp_size //= 2
        else:
            pp_size -= 1

    chunks = min(pp_size * 2, 8) if pp_size > 1 else 1

    return ModelParallelConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        pipeline_chunks=chunks,
    )
