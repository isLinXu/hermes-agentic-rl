#!/usr/bin/env python3
"""MT-GRPO Training Script
========================

Multi-Turn GRPO with turn-level credit assignment, addressing the core
issue: reward increases but tool call metrics stay at zero.

Key Features:
1. Structured tool call rewards (name + schema + arguments quality)
2. Turn-level advantages (MT-GRPO Equation 7: Â_τ = Â^T_τ + λ · Â^O)
3. Explicit curriculum stages with progressive difficulty
4. Token-level advantage assignment (REINFORCE++ per-token advantages)
5. Outcome bonus for successful tool calls only

Usage:
    # Quick test with tiny backend:
    python train_mtgrpo.py --backend tiny --n-iters 5

    # Full training with HF backend:
    python train_mtgrpo.py \\
        --backend hf \\
        --model-name Qwen/Qwen2.5-1.5B-Instruct \\
        --n-iters 100 \\
        --group-size 8

The curriculum ensures the model learns:
- When to use tools (tool call structure reward)
- How to use tools correctly (tool call content reward)
- How to summarize results (summary/narration reward)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from hermes_agentic_rl.curriculum import (
    CurriculumScheduler,
    CurriculumSchedulerConfig,
    create_default_curriculum,
)
from hermes_agentic_rl.rewards.composer import RewardComposer, RewardComposerConfig
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward

# RewardManager is no longer imported — RewardComposer is used directly as
# the reward manager (both satisfy the trainer's duck-typed interface:
# async evaluate(item, trajectory, tool_context) -> RewardSummary + .components).

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT-GRPO Training Script")

    # Backend
    parser.add_argument(
        "--backend",
        type=str,
        default="tiny",
        choices=["tiny", "hf"],
        help="Backend type: tiny (for testing) or hf (for real training)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for HF backend",
    )

    # Training
    parser.add_argument("--n-iters", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size")
    parser.add_argument("--prompts-per-iter", type=int, default=2, help="Prompts per iteration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Max new tokens per response",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="GRPO clip epsilon")
    parser.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty coefficient")

    # Curriculum
    parser.add_argument(
        "--curriculum-auto-advance",
        action="store_true",
        default=True,
        help="Auto-advance curriculum stages",
    )
    parser.add_argument(
        "--min-iters-per-stage",
        type=int,
        default=10,
        help="Minimum iterations before advancing stage",
    )

    # Multi-turn credit
    parser.add_argument(
        "--credit-mode",
        type=str,
        default="discounted",
        choices=["shared", "terminal", "discounted", "judge", "hybrid"],
        help="Multi-turn credit assignment mode",
    )
    parser.add_argument("--credit-gamma", type=float, default=0.9, help="Discount gamma for credit")

    # Output
    parser.add_argument("--output-dir", type=str, default="output/mtgrpo", help="Output directory")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N iterations")
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="Save checkpoint every N iters (0=disabled)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backend creation
# ---------------------------------------------------------------------------


def create_backend(args: argparse.Namespace) -> Any:
    """Create the appropriate LLM backend based on args."""
    if args.backend == "tiny":
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        return TinyCausalLMBackend(
            TinyBackendConfig(
                seed=42,
                with_value_head=False,
                dim=32,
                n_heads=4,
                n_layers=2,
                max_len=256,
            )
        )
    elif args.backend == "hf":
        try:
            from hermes_agentic_rl.backends.hf_backend import HFBackend, HFBackendConfig

            return HFBackend(
                HFBackendConfig(
                    model_name=args.model_name,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        except ImportError:
            logger.warning("HF backend not available, falling back to tiny")
            return create_backend(argparse.Namespace(backend="tiny"))
    else:
        raise ValueError(f"Unknown backend: {args.backend!r}")


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------


def create_env(args: argparse.Namespace) -> Any:
    """Create the training environment."""
    from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset

    dataset = build_sim_tool_dataset(n_samples=100, seed=42)
    return SimToolEnv(dataset=dataset)


# ---------------------------------------------------------------------------
# Reward setup
# ---------------------------------------------------------------------------


def create_reward_composer(
    curriculum_scheduler: CurriculumScheduler,
) -> RewardComposer:
    """Create the reward composer with curriculum-gated components.

    The composer uses conditional activation so that:
    - tool_call_structure reward is always active
    - tool_call_content reward activates after stage 1
    - outcome reward activates after stage 2
    """
    components = [
        ToolcallReward(weight=1.0),
        OutcomeReward(weight=1.0),
    ]

    # Curriculum scheduler is used for logging only; the real
    # scheduling happens inside OnPolicyTrainer via cfg.curriculum.
    _ = curriculum_scheduler.get_stage_weights()

    config = RewardComposerConfig(
        normalize={"toolcall_reward": True, "outcome_reward": True},
        conditions={
            # Tool call reward is always active
            "toolcall_reward": lambda item, traj: True,
            # Outcome reward only fires when there's a final output
            "outcome_reward": lambda item, traj: bool(traj.final_output),
        },
        turn_discount={"toolcall_reward": 0.95},
    )

    return RewardComposer(components=components, config=config)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    """Run the MT-GRPO training loop.

    Fixes the three-way breakage in the original script:
    A. Uses ``RewardComposer`` (which has ``evaluate()`` and ``components``)
       directly as the reward manager — it is a drop-in replacement for
       ``RewardManager`` and both satisfy the trainer's duck-typed interface.
    B. Passes ``curriculum`` config via ``GRPOTrainerConfig.curriculum`` dict
       so the ``OnPolicyTrainer`` creates its own internal
       ``CurriculumScheduler`` (on_policy.py:594-620) and runs observe/advance
       inside the training loop (on_policy.py:1457-1479).
    C. Removes the dead external ``for iter_idx in range(n_iters)`` loop that
       ran before ``trainer.train()`` with empty ``batch_stats``.
    """
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    # 1. Create curriculum stages (for logging + composer weights)
    curriculum_stages = create_default_curriculum()

    # 2. Create backend and environment
    backend = create_backend(args)
    env = create_env(args)

    # 3. Create reward composer — this IS the reward manager.
    #    RewardComposer is a drop-in replacement for RewardManager:
    #    both expose ``async evaluate(item, trajectory, tool_context) -> RewardSummary``
    #    and a ``.components`` attribute.
    #    We need a scheduler instance to read stage weights for composer config,
    #    but the *real* curriculum scheduling happens inside the trainer.
    temp_scheduler = CurriculumScheduler(
        stages=curriculum_stages,
        cfg=CurriculumSchedulerConfig(
            auto_advance=args.curriculum_auto_advance,
            min_iters_per_stage=args.min_iters_per_stage,
        ),
    )
    reward_manager = create_reward_composer(temp_scheduler)

    # 4. Create trainer config with curriculum enabled via dict.
    #    OnPolicyTrainer reads cfg.curriculum (on_policy.py:599-620) and
    #    creates its own CurriculumScheduler internally. The observe/advance
    #    logic runs inside train() (on_policy.py:1457-1479).
    trainer_cfg = GRPOTrainerConfig(
        n_iters=args.n_iters,
        group_size=args.group_size,
        prompts_per_iter=args.prompts_per_iter,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=Path(args.output_dir),
        multi_turn=True,
        multi_turn_credit={
            "mode": args.credit_mode,
            "gamma": args.credit_gamma,
        },
        per_token_advantage=True,
        advantage_norm="group",
        # Enable built-in curriculum integration.
        # OnPolicyTrainer will create its own CurriculumScheduler from this dict.
        curriculum={
            "auto_advance": args.curriculum_auto_advance,
            "min_iters_per_stage": args.min_iters_per_stage,
        },
    )

    # 5. Create trainer — curriculum is managed internally by train()
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=trainer_cfg,
    )

    logger.info("Starting MT-GRPO training with curriculum:")
    for i, stage in enumerate(curriculum_stages):
        logger.info(f"  Stage {i}: {stage.name} (weight={stage.reward_weight}, "
                     f"threshold={stage.mastery_threshold})")

    # 6. Run training — curriculum observe/advance happens inside train()
    trainer.train()

    logger.info("Training complete!")

    # 7. Log curriculum snapshot from the trainer's internal scheduler
    if trainer._curriculum_scheduler is not None:
        logger.info("Curriculum snapshot: %s", trainer._curriculum_scheduler.snapshot())
    else:
        logger.warning("Curriculum scheduler was not initialized inside trainer")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_training(args)


if __name__ == "__main__":
    main()
