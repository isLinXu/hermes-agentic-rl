#!/usr/bin/env python
"""SFT + RL Hybrid Training Example.

Demonstrates two built-in SFT capabilities of hermes-agentic-rl:

1. **Bootstrap SFT** — Run N rounds of supervised fine-tuning *before* RL
   to give the policy a warm start. Useful when the base model has never
   seen the target task format.

2. **Interleaved SFT** — Insert SFT rounds *between* RL iterations at a
   fixed interval. Prevents catastrophic forgetting and stabilises
   training when the reward signal is sparse.

Both features are built into ``OnPolicyTrainer`` and triggered purely
via ``GRPOTrainerConfig`` fields — no extra code needed.

Usage::

    python examples/sft_plus_rl.py            # quick smoke test (TinyBackend)
    python examples/sft_plus_rl.py --real     # real model (needs GPU)

The corresponding YAML config is ``configs/sft_plus_rl.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

logger = logging.getLogger(__name__)


def run_sft_plus_rl(use_real_model: bool = False) -> None:
    """Run SFT+RL hybrid training.

    Parameters
    ----------
    use_real_model : bool
        If True, use a HuggingFace backend (requires GPU).
        If False, use TinyBackend for quick smoke testing.
    """
    # ── Backend ──
    if use_real_model:
        from hermes_agentic_rl.backends.hf_backend import HFBackend, HFBackendConfig

        backend = HFBackend(
            HFBackendConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                max_new_tokens=256,
            )
        )
    else:
        backend = TinyCausalLMBackend(
            TinyBackendConfig(
                seed=42,
                with_value_head=True,
                dim=32,
                n_heads=4,
                n_layers=2,
                max_len=256,
            )
        )

    # ── Environment ──
    env = EchoTaskEnv(build_default_echo_dataset())

    # ── Reward ──
    reward_manager = RewardManager(rewards=[EchoRewardComponent(weight=1.0)])

    # ── Config: SFT + RL ──
    cfg = GRPOTrainerConfig(
        n_iters=10,
        group_size=4,
        prompts_per_iter=2,
        lr=1e-3,
        max_new_tokens=32,
        temperature=1.0,
        log_every=1,
        seed=42,
        output_dir="output/sft_plus_rl_demo",
        # ── Bootstrap SFT: warm-start before RL ──
        bootstrap_sft_rounds=3,       # 3 rounds of SFT before iteration 0
        bootstrap_sft_samples=16,     # 16 supervised samples per round
        bootstrap_sft_lr=5e-2,        # higher LR for fast warm-start
        # ── Interleaved SFT: stabilise during RL ──
        interleave_sft_every=2,       # SFT every 2 RL iterations
        interleave_sft_samples=8,     # 8 supervised samples per SFT round
        interleave_sft_lr=1e-3,       # gentler LR during RL
    )

    # ── Train ──
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=cfg,
    )

    logger.info("Starting SFT+RL training...")
    logger.info("  Bootstrap: %d rounds, %d samples/round, lr=%g",
                cfg.bootstrap_sft_rounds, cfg.bootstrap_sft_samples, cfg.bootstrap_sft_lr)
    logger.info("  Interleave: every %d iters, %d samples, lr=%g",
                cfg.interleave_sft_every, cfg.interleave_sft_samples, cfg.interleave_sft_lr)

    stats = trainer.train()

    # ── Report ──
    logger.info("Training complete! %d iterations", len(stats.iters))
    for i, iter_stats in enumerate(stats.iters):
        sft_info = ""
        if "sft_loss" in iter_stats:
            sft_info = f"  sft_loss={iter_stats['sft_loss']:.4f}"
        if "n_sft_samples" in iter_stats:
            sft_info += f"  n_sft={iter_stats['n_sft_samples']}"
        logger.info("  iter %d: reward=%.4f%s", i, iter_stats.get("mean_reward", 0), sft_info)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="SFT+RL hybrid training demo")
    parser.add_argument(
        "--real", action="store_true",
        help="Use real HuggingFace model instead of TinyBackend"
    )
    args = parser.parse_args()

    run_sft_plus_rl(use_real_model=args.real)


if __name__ == "__main__":
    main()
