# Model Parallelism Technical Assessment Report

**Project:** hermes-agentic-rl v0.12.0
**Module:** `hermes_agentic_rl.distributed.model_parallel`
**Date:** 2026-07-08
**Scope:** Tensor Parallelism (TP), Pipeline Parallelism (PP), Hybrid 3D Parallelism, Expert Parallelism (EP), and third-party integration hooks (Megatron / DeepSpeed).

---

## 1. Executive Summary

The `model_parallel.py` module provides a **clean, well-structured interface** for model parallelism but is currently **interface-complete / implementation-stub** for all non-trivial strategies. The abstraction layer (`ModelParallelConfig`, `ModelParallelStrategy`, `apply_model_parallel`) is production-ready. The concrete sharding implementations (`TensorParallelStrategy`, `PipelineParallelStrategy`, `HybridParallelStrategy`) are **placeholders** that return the model unchanged and only perform no-op gradient/state-dict operations.

This assessment documents the exact gap between the interface and production-grade execution, and provides a prioritized roadmap to close it.

---

## 2. Current Implementation Status (Per Strategy)

### 2.1 `_NoOpStrategy` — ✅ Complete
**Status:** Production-ready.
**What it does:** Pass-through when `tensor_parallel_size == 1`, `pipeline_parallel_size == 1`, and `expert_parallel_size == 1`. Correctly returns `model.state_dict()` and skips gradient synchronization.

### 2.2 `TensorParallelStrategy` — ⚠️ Interface-Complete / Stub Implementation
**Status:** Partial. The control flow (backend selection, lazy imports, fallback) is well-designed. The actual layer sharding is **not implemented**.

**What works today:**
- `apply()` correctly detects when torch.distributed is uninitialized and falls back to no-op.
- Backend selection logic (`auto` → megatron → torch) is sound.
- `sync_gradients()` implements a naive all-reduce over the **global** world size, which is functionally correct only for a single TP group.

**What is missing / broken:**
- **No actual layer sharding.** `_try_megatron()` and `_try_torch_native()` import the target libraries but return `model` unchanged. No `nn.Linear` layers are replaced with column-wise or row-wise parallel counterparts.
- **No TP process group creation.** `sync_gradients()` uses `dist.all_reduce()` over the global default group instead of the TP-specific subgroup. In a hybrid (TP+PP) setup this will corrupt gradients by synchronizing across PP boundaries.
- **State-dict gather is incorrect.** `gather_state_dict()` does `dist.all_gather()` into a list, then returns `gathered[dist.get_rank()]`, which is just the local shard. It does **not** concatenate along the sharding dimension. The comment acknowledges this is a placeholder, but the code silently returns wrong data rather than raising.
- **No column-wise / row-wise tracking.** Megatron TP requires distinguishing whether a linear layer is sharded on the output dimension (column-wise) or input dimension (row-wise). There is no metadata tracking per parameter.

**Verdict:** ~20% production-ready. The scaffolding is solid; the sharding logic is absent.

### 2.3 `PipelineParallelStrategy` — ⚠️ Interface-Complete / Stub Implementation
**Status:** Partial. The wrapping hook exists but performs no actual stage splitting.

**What works today:**
- `apply()` correctly detects when PP is disabled and falls back.
- `sync_gradients()` is correctly a no-op with the comment noting that PP schedulers handle this automatically.

**What is missing / broken:**
- **No stage splitting.** `_try_torch_pipeline()` returns the model unchanged. It does not use `torch.distributed.pipeline.sync.Pipe`, nor does it partition `nn.Sequential` or `nn.ModuleList` across ranks.
- **No micro-batch scheduling.** `pipeline_chunks` is stored in config but never used. There is no `PipelineEngine` or forward/backward scheduling.
- **State-dict gather is semantically wrong for PP.** `gather_state_dict()` broadcasts every tensor from rank 0, but in a real PP setup each rank owns a disjoint subset of layers. The current code would overwrite every rank's layers with rank 0's copy.
- **No inter-stage communication.** `send`/`recv` between PP ranks is not implemented.

**Verdict:** ~15% production-ready. The config and lifecycle hooks are present, but the core pipeline engine is missing.

### 2.4 `HybridParallelStrategy` — ⚠️ Composition Stub with Active Bug
**Status:** Partial. It correctly chains TP and PP strategies, but contains a **known runtime bug** that was present before this assessment and has now been fixed.

**What works today:**
- `apply()` correctly delegates to `TensorParallelStrategy` then `PipelineParallelStrategy`, which is the correct 3D ordering (TP intra-node, PP inter-node).
- `sync_gradients()` correctly delegates to TP only (PP handles itself).
- `teardown()` correctly delegates to both sub-strategies.

**What was broken (pre-assessment):**
- `gather_state_dict()` called `self._pp_strategy.gather_state_dict_from_state(state)`, a method that **does not exist** on `PipelineParallelStrategy`. This would raise `AttributeError` at runtime if `HybridParallelStrategy` were ever used.

**Fix applied:**
- `gather_state_dict()` now returns the TP-gathered state directly, with a comment explaining that a full PP gather would need to collect stage-specific layers across pipeline ranks. This is a safe no-op fallback because both strategies currently return the model unchanged.

**What is missing:**
- **No DP integration.** The docstring claims TP + PP + DP, but the class only composes TP and PP. The `DistributedConfig` (`dp_cfg`) is stored but never used. FSDP/DDP wrapping is expected to be done by the caller, which is acceptable but should be documented more explicitly.
- **No cross-group coordination.** In real 3D parallelism, you need separate process groups for TP, PP, and DP. `HybridParallelStrategy` does not create or manage these groups.

**Verdict:** ~25% production-ready (scaffolding is good, bug fixed, but no actual 3D group management).

### 2.5 `compute_parallel_config()` — ✅ Heuristic Complete (No EP)
**Status:** The heuristic function works correctly for TP and PP sizing. It does not account for `expert_parallel_size` (EP), which was added during this assessment.

**Note:** EP is a MoE-specific optimization. The heuristic should eventually consider EP when the model contains sparse expert layers, but this is a future enhancement.

---

## 3. What Works Today (No External Dependencies)

| Capability | Status | Notes |
|---|---|---|
| Single-GPU training | ✅ Full | `_NoOpStrategy` passes model through unchanged. |
| DDP / FSDP rollouts | ✅ Full | `distributed.py` (separate module) handles data parallelism. `model_parallel.py` is explicitly a no-op when disabled. |
| Config validation | ✅ Full | `ModelParallelConfig.validate()` catches invalid sizes and backend mismatches. |
| Heuristic sizing | ✅ Full | `compute_parallel_config()` recommends TP/PP split given model size and GPU count. |
| Lazy imports | ✅ Full | All strategies gracefully degrade when `torch.distributed` or third-party libs are missing. |

---

## 4. What Requires Megatron / DeepSpeed Integration

| Capability | Missing Integration | Required Dependency | Why It Can't Be Done in Pure PyTorch |
|---|---|---|---|
| **Tensor Parallelism** | Megatron `ColumnParallelLinear`, `RowParallelLinear`, `tensor_parallel` module | `megatron-core` | PyTorch native TP (`torch.distributed.tensor.parallel`) is experimental and lacks mature column/row sharding with fused all-reduce for linear layers. Megatron is the industry standard. |
| **Pipeline Parallelism** | Megatron `get_forward_backward_func` or DeepSpeed `PipelineEngine` | `megatron-core` or `deepspeed` | PyTorch's built-in `torch.distributed.pipeline.sync.Pipe` is deprecated in favor of newer APIs but still lacks robust pipeline scheduling, bubble optimization, and activation checkpointing integration for large models. |
| **3D Parallelism (TP+PP+DP)** | Megatron `parallel_state` (process groups) or DeepSpeed `DeepSpeedEngine` + `PipelineEngine` | `megatron-core` or `deepspeed` | Pure PyTorch requires manual management of 3 process-group types (TP, PP, DP) and careful synchronization ordering. Megatron's `parallel_state` abstracts this. |
| **ZeRO-1/2/3** | DeepSpeed `DeepSpeedEngine` | `deepspeed` | ZeRO shards optimizer states and gradients across DP ranks, which is orthogonal to TP/PP but essential for fitting large models. Hermes does not currently implement ZeRO. |
| **Expert Parallelism (MoE)** | Megatron `ExpertParallel` or DeepSpeed MoE | `megatron-core` or `deepspeed` | EP requires all-to-all communication between expert shards. No PyTorch-native API exists for this. |
| **State-Dict Gather** | Megatron `gather_from_tensor_model_parallel_region` or custom `all_gather` + concat | `megatron-core` or custom impl | Need to know the sharding dimension per tensor to concatenate correctly. |

---

## 5. Recommended Path Forward

### Phase 1 — Foundation (2–3 weeks)
**Goal:** Make `TensorParallelStrategy` production-ready for single-node, multi-GPU training.

1. **Adopt `MegatronIntegration` hooks.**
   - Implement `configure_megatron_tp()` to replace `nn.Linear` with `ColumnParallelLinear` / `RowParallelLinear` based on layer position (e.g., MLP output is column-wise, next MLP input is row-wise).
   - Create TP process groups via `torch.distributed.new_group()` and integrate with Megatron's `parallel_state` if available, or fallback to custom groups.
2. **Fix `sync_gradients()` to target TP group only.**
3. **Implement correct `gather_state_dict()` with per-parameter shard metadata.**
   - Add a `_shard_map: dict[str, dict]` to `TensorParallelStrategy` that tracks `dim` and `world_size` for each parameter.
   - Use `torch.cat(gathered, dim=shard_dim)` instead of returning the local rank's tensor.
4. **Add unit tests with `torch.distributed.launch` on 2-GPU mock.**

### Phase 2 — Pipeline (2–3 weeks)
**Goal:** Make `PipelineParallelStrategy` production-ready for multi-node training.

1. **DeepSpeed Pipeline first (lower effort).**
   - `DeepSpeedIntegration.configure_deepspeed_pipeline()` can wrap a model with `deepspeed.PipelineEngine` given a stage partition function.
   - DeepSpeed handles the micro-batch scheduling, bubble reduction, and gradient accumulation automatically.
2. **Megatron Pipeline as alternative.**
   - Megatron's PP is more flexible for custom model architectures but requires manual stage splitting and `forward_backward_func` scheduling.
3. **Implement stage splitting utility.**
   - Add a `partition_model_into_stages(model, n_stages)` helper that splits `nn.Sequential` or `nn.ModuleList` blocks evenly by parameter count.
4. **Fix `gather_state_dict()` to collect disjoint layers from each PP rank.**
   - Use `dist.gather()` to rank 0 or `all_gather_object` for layer name lists.

### Phase 3 — 3D Hybrid (1–2 weeks)
**Goal:** Connect TP + PP + DP with correct process-group topology.

1. **Implement process-group factory.**
   - Given `tp_size`, `pp_size`, `dp_size`, create 3 orthogonal groups per rank.
   - Map each rank to its `(tp_rank, pp_rank, dp_rank)` coordinate.
2. **Update `HybridParallelStrategy` to pass group handles to sub-strategies.**
3. **Integration test:** run a 7B-parameter model on 8 GPUs (2 TP × 2 PP × 2 DP).

### Phase 4 — Expert Parallelism (1–2 weeks)
**Goal:** Support MoE models.

1. **Add `ExpertParallelStrategy` or extend `TensorParallelStrategy`.**
   - EP is logically similar to TP but requires `all_to_all` instead of `all_reduce` / `all_gather`.
2. **Integrate Megatron MoE or DeepSpeed MoE wrappers.**
3. **Update `compute_parallel_config()` heuristic to consider EP when model has sparse layers.**

---

## 6. Estimated Effort to Production-Ready

| Parallelism Type | Current State | Target State | Estimated Effort | Risk Level |
|---|---|---|---|---|
| **Tensor Parallelism** | 20% (stub) | 90% (Megatron-backed) | 2–3 weeks | Low — Megatron API is stable. |
| **Pipeline Parallelism** | 15% (stub) | 80% (DeepSpeed-backed) | 2–3 weeks | Medium — stage splitting is architecture-dependent. |
| **Hybrid 3D (TP+PP+DP)** | 25% (composition stub) | 85% (group-aware) | 1–2 weeks | Low — mostly wiring once TP/PP work. |
| **ZeRO (DP optimizer sharding)** | 0% (not implemented) | 80% (DeepSpeed-backed) | 1–2 weeks | Low — DeepSpeed handles this. |
| **Expert Parallelism (MoE)** | 10% (config field only) | 70% (Megatron/DeepSpeed) | 2–3 weeks | High — MoE architectures vary widely. |

**Total estimate:** 8–13 weeks of focused engineering to reach a production-grade 3D-parallel training stack for models up to 70B parameters on 64+ GPUs.

---

## 7. New Integration Hooks Added (This Assessment)

To close the dependency-gap and provide a clean upgrade path, the following integration classes were added to `model_parallel.py`:

### `MegatronIntegration`
- `try_megatron_import()` — lazy import guard for `megatron.core`.
- `configure_megatron_tp(model, tp_size)` — sets up Megatron tensor parallelism when the library is available; returns `None` otherwise (graceful degradation).
- `configure_megatron_pp(model, pp_size, num_chunks)` — sets up Megatron pipeline parallelism when available; returns `None` otherwise.

### `DeepSpeedIntegration`
- `try_deepspeed_import()` — lazy import guard for `deepspeed`.
- `configure_deepspeed_zero(model, zero_stage)` — sets up DeepSpeed ZeRO-1/2/3 when available.
- `configure_deepspeed_pipeline(model, pp_size, num_chunks)` — sets up DeepSpeed `PipelineEngine` when available.

Both classes follow the existing lazy-import pattern: they are **no-ops when dependencies are missing**, allowing the codebase to be installed and run without Megatron or DeepSpeed, while enabling a single-line upgrade path (`pip install megatron-core` or `pip install deepspeed`) once the production sharding logic is implemented.

### `ModelParallelConfig` Enhancements
- Added `expert_parallel_size: int = 1` to support MoE sharding in the configuration layer.
- Updated `enabled` and `total_parallel_size` to include EP.
- Added validation for `expert_parallel_size >= 1`.

### Bug Fixes
- **Fixed `HybridParallelStrategy.gather_state_dict()`** — removed the non-existent `gather_state_dict_from_state()` call on `PipelineParallelStrategy` and replaced it with a safe fallback that returns the TP-gathered state, noting that full PP gather is pending production implementation.

---

## 8. Known Technical Debt

1. **State-dict gather in `TensorParallelStrategy` is silently incorrect.** It does not concatenate shards. This should be fixed as soon as actual TP sharding is implemented, or it should raise `NotImplementedError` in the interim to prevent silent data corruption.
2. **State-dict gather in `PipelineParallelStrategy` broadcasts rank 0's state to all ranks.** In a real PP deployment, this is wrong because each rank owns a different subset of layers. Should be fixed alongside the PP engine implementation.
3. **`sync_gradients()` in `TensorParallelStrategy` uses the global world.** It should use the TP-specific subgroup. This is a correctness bug in a multi-parallel setup.
4. **`compute_parallel_config()` does not consider EP.** When MoE is used, the heuristic may under-utilize GPUs or recommend an invalid configuration. Add EP awareness once `ExpertParallelStrategy` is implemented.
5. **No MoE-specific strategy class.** `expert_parallel_size` is accepted in config but there is no `ExpertParallelStrategy` class. This should be added in Phase 4.

---

## 9. Conclusion

The `model_parallel.py` module is a **well-designed interface** with solid configuration, fallback logic, and lifecycle hooks. It is currently a **scaffold** rather than a production engine. The addition of `MegatronIntegration` and `DeepSpeedIntegration` hooks provides the necessary upgrade path. The recommended next step is **Phase 1: Tensor Parallelism**, because TP is the most self-contained (single-node, no inter-node communication), has the clearest industry-standard implementation (Megatron), and provides the highest immediate value for fitting larger models on single-node multi-GPU workstations.

**Priority order:**
1. Fix TP state-dict gather and gradient sync (correctness).
2. Implement Megatron-backed TP sharding (Phase 1).
3. Implement DeepSpeed-backed PP (Phase 2).
4. Wire 3D process groups (Phase 3).
5. Add MoE / EP support (Phase 4).
