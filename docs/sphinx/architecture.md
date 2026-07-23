# Architecture

## Overview

```
┌─────────────────────────────────────────────────────┐
│                   YAML Config                        │
│  (validated by Pydantic / fallback validator)        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              TrainingOrchestrator                     │
│  Builds backend, env, rewards, trainer from config   │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│  OnPolicy    │ │  Reward    │ │   Rollout    │
│  Trainer     │ │  Manager   │ │   Pool       │
│  (GRPO/PPO)  │ │  (Composer)│ │  (vLLM/MP)   │
└──────┬───────┘ └────────────┘ └──────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                  Algorithm Layer                      │
│  GRPOAlgo | PPOAlgo | HybridAlgo | DAPOAlgo          │
│  + StalenessAdaptiveTIS + OPD + Replay Buffer        │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│  LoRA Hot    │ │  Client-   │ │  Quantized   │
│  Reload Mgr  │ │  Server    │ │  Backend     │
└──────────────┘ └────────────┘ └──────────────┘
```

## Key Components

### OnPolicyTrainer

The base training loop (`hermes_agentic_rl.trainers.on_policy.OnPolicyTrainer`)
implements the standard RL loop:

1. Collect group rollouts (G samples per prompt)
2. Evaluate rewards (via RewardManager or RewardComposer)
3. Compute loss (via algorithm: GRPO, PPO, Hybrid)
4. Optimizer step (with gradient accumulation, clipping, AMP)
5. Optional: KL update, checkpoint, eval, curriculum advance

### Reward System

- **RewardManager**: flat-list aggregator (simple weighted sum)
- **RewardComposer**: stateful aggregator with running normalization,
  conditional gating, turn discounting, and parallel evaluation
- **RULER**: rule-based reward extraction with built-in templates
  (exact_match, contains, regex, length, etc.)

### Rollout Architecture

- **BatchRolloutGenerator**: sequential rollout with the policy backend
- **MPRolloutPool**: multi-process rollout pool with weight broadcasting
- **FaultTolerantRolloutPool**: elastic pool with health checks and
  automatic worker replacement
- **Client-Server**: decoupled client/server with version-tagged weight
  sync (OpenPipe/ART style)

### Advanced Features

- **StalenessAdaptiveTIS**: dynamically adjusts TIS rho_clip based on
  observed rollout staleness in pipelined/replay mode
- **LoRAHotReloadManager**: merges LoRA adapter deltas into shadow base
  weights and syncs to vLLM without full weight copy
- **QuantizedRolloutBackend**: GPTQ/AWQ/GGUF inference via vLLM or llama.cpp
- **ModelParallelStrategy**: interface for TP/PP/EP across backends

## Data Flow

```
Prompt → Policy.generate() → Trajectory
                                │
                    RewardManager.evaluate()
                                │
                    Algo.compute_loss()
                                │
                    Optimizer.step()
                                │
                    ┌── Checkpoint (async)
                    ├── Curriculum advance
                    ├── Eval hook
                    └── LoRA hot-reload sync
```
