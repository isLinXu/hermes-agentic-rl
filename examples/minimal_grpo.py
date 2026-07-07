#!/usr/bin/env python
"""Minimal GRPO Training Example — 5 minute quickstart.

Trains a tiny model on arithmetic tasks using GRPO. No GPU needed.

This is the simplest possible end-to-end training script:

1. Create a TinyBackend (CPU, < 1MB model)
2. Create a SimToolEnv (arithmetic problems)
3. Create a RewardManager (checks if answer is correct)
4. Create a GRPOTrainer with a minimal config
5. Call trainer.train()

Usage::

    python examples/minimal_grpo.py
"""

from __future__ import annotations

import logging
import sys

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # 1. Backend — tiny model on CPU
    backend = TinyCausalLMBackend(
        TinyBackendConfig(
            seed=42,
            with_value_head=True,  # GRPO needs value head for advantage estimation
            dim=32,
            n_heads=4,
            n_layers=2,
            max_len=128,
        )
    )

    # 2. Environment — simple arithmetic
    dataset = build_sim_tool_dataset(n=50, seed=42)
    env = SimToolEnv(dataset=dataset)

    # 3. Reward — did the model produce the correct answer?
    reward_manager = RewardManager(rewards=[OutcomeReward(weight=1.0)])

    # 4. Config — minimal GRPO
    cfg = GRPOTrainerConfig(
        n_iters=5,
        group_size=4,
        prompts_per_iter=2,
        lr=1e-3,
        max_new_tokens=32,
        temperature=1.0,
        log_every=1,
        seed=42,
        output_dir="output/minimal_grpo_demo",
    )

    # 5. Train
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=cfg,
    )

    logger.info("Starting minimal GRPO training...")
    stats = trainer.train()

    logger.info("Done! %d iterations completed.", len(stats.iters))
    for i, s in enumerate(stats.iters):
        logger.info("  iter %d: mean_reward=%.4f", i, s.get("mean_reward", 0))


if __name__ == "__main__":
    main()
