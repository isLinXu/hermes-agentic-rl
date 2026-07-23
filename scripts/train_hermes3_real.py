"""Standalone training script for agentlans/NousResearch-Hermes-3-Dataset."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_agentic_rl.backends.tiny import (  # noqa: E402
    TinyBackendConfig,
    TinyCausalLMBackend,
)
from hermes_agentic_rl.core.reward_manager import RewardManager  # noqa: E402
from hermes_agentic_rl.envs.hermes3_dataset_env import Hermes3DatasetEnv  # noqa: E402
from hermes_agentic_rl.rewards.similarity_reward import (  # noqa: E402
    CompositeSimilarityReward,
)
from hermes_agentic_rl.trainers.grpo_trainer import (  # noqa: E402
    GRPOTrainer,
    GRPOTrainerConfig,
)

logger = logging.getLogger("train_hermes3_real")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train hermes-agentic-rl on NousResearch-Hermes-3-Dataset"
    )
    p.add_argument("--dataset-name", default="agentlans/NousResearch-Hermes-3-Dataset")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no-streaming", action="store_false", dest="streaming")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["tiny", "transformers"], default="tiny")
    p.add_argument("--model-name", default="microsoft/DialoGPT-small")
    p.add_argument("--device", default="auto")
    p.add_argument("--tiny-dim", type=int, default=256)
    p.add_argument("--tiny-n-heads", type=int, default=8)
    p.add_argument("--tiny-n-layers", type=int, default=6)
    p.add_argument("--tiny-max-len", type=int, default=512)
    p.add_argument("--n-iters", type=int, default=20)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--prompts-per-iter", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--minibatch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--bootstrap-sft-rounds", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument(
        "--reward-metric",
        choices=[
            "rouge_l", "rouge_1", "rouge_2", "bleu",
            "exact_match", "token_overlap", "composite",
        ],
        default="composite",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--tensorboard-dir", default="./runs/hermes3_real")
    p.add_argument("--output-dir", default="./outputs/hermes3_real")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Hermes-Agentic-RL: Real Training on Hermes-3-Dataset")
    logger.info("=" * 60)
    logger.info("Dataset: %s (split=%s)", args.dataset_name, args.dataset_split)
    logger.info("Backend: %s", args.backend)
    logger.info("Iters: %d, Group: %d", args.n_iters, args.group_size)
    logger.info("Limit: %s", args.limit if args.limit else "full")

    # 1. Build backend
    if args.backend == "tiny":
        backend = TinyCausalLMBackend(
            TinyBackendConfig(
                dim=args.tiny_dim,
                n_heads=args.tiny_n_heads,
                n_layers=args.tiny_n_layers,
                max_len=args.tiny_max_len,
                seed=args.seed,
                with_value_head=False,
            )
        )
        logger.info(
            "Tiny: dim=%d heads=%d layers=%d",
            args.tiny_dim, args.tiny_n_heads, args.tiny_n_layers,
        )
    else:
        from hermes_agentic_rl.backends.hf import HFCausalLMBackend, HFBackendConfig

        backend = HFCausalLMBackend(
            HFBackendConfig(
                model_name_or_path=args.model_name,
                device=args.device,
            )
        )
        logger.info("HF: %s on %s", args.model_name, args.device)

    # 2. Build environment (async setup in sync context)
    logger.info("Loading dataset...")

    async def _setup_env() -> Hermes3DatasetEnv:
        env = Hermes3DatasetEnv.from_hf_dataset(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            limit=args.limit,
            shuffle=True,
            seed=args.seed,
            streaming=args.streaming,
            max_prompt_chars=2048,
        )
        await env.setup()
        return env

    env = asyncio.run(_setup_env())
    logger.info("Environment: %d items", len(env.items))

    # 3. Build reward
    if args.reward_metric == "composite":
        reward = CompositeSimilarityReward(
            metrics=[("rouge_l", 0.5), ("bleu", 0.3), ("exact_match", 0.2)]
        )
        logger.info("Reward: CompositeSimilarity")
    else:
        from hermes_agentic_rl.rewards.similarity_reward import SimilarityReward

        reward = SimilarityReward(metric=args.reward_metric)
        logger.info("Reward: %s", args.reward_metric)

    reward_manager = RewardManager([reward])

    # 4. Build trainer
    cfg = GRPOTrainerConfig(
        n_iters=args.n_iters,
        group_size=args.group_size,
        prompts_per_iter=args.prompts_per_iter,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        max_new_tokens=args.max_seq_len,
        bootstrap_sft_rounds=args.bootstrap_sft_rounds,
        grad_clip=args.grad_clip,
        checkpoint_every=args.checkpoint_every,
        output_dir=Path(args.output_dir),
    )

    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=cfg,
    )

    logger.info("Starting training...")
    stats = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Stats: %s", stats)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
