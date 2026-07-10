"""Standalone training script using local cached dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_agentic_rl.backends.tiny import (
    TinyBackendConfig,
    TinyCausalLMBackend,
)
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.hermes3_dataset_env import Hermes3DatasetEnv
from hermes_agentic_rl.rewards.similarity_reward import CompositeSimilarityReward
from hermes_agentic_rl.trainers.grpo_trainer import (
    GRPOTrainer,
    GRPOTrainerConfig,
)

logger = logging.getLogger("train_local")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with local cached dataset")
    p.add_argument("--json-path", default="data/hermes3_100.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["tiny", "transformers"], default="tiny")
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tiny-dim", type=int, default=256)
    p.add_argument("--tiny-n-heads", type=int, default=8)
    p.add_argument("--tiny-n-layers", type=int, default=6)
    p.add_argument("--tiny-max-len", type=int, default=512)
    p.add_argument("--n-iters", type=int, default=5)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--prompts-per-iter", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--minibatch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=32)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bootstrap-sft-rounds", type=int, default=0)
    p.add_argument("--bootstrap-sft-samples", type=int, default=32)
    p.add_argument("--bootstrap-sft-lr", type=float, default=1e-4)
    p.add_argument("--bootstrap-sft-epochs", type=int, default=1)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--output-dir", default="./outputs/local_train")
    p.add_argument("--save-stats", default="stats.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Hermes-Agentic-RL: Local Dataset Training")
    logger.info("Backend: %s", args.backend)
    logger.info("JSON: %s", args.json_path)
    if args.bootstrap_sft_rounds > 0:
        logger.info("SFT warm-start: %d rounds, %d samples", args.bootstrap_sft_rounds, args.bootstrap_sft_samples)
    logger.info("=" * 60)

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
        logger.info("Tiny backend: dim=%d", args.tiny_dim)
    else:
        from hermes_agentic_rl.backends.hf import HFCausalLMBackend, HFBackendConfig

        backend = HFCausalLMBackend(
            HFBackendConfig(
                model_name_or_path=args.model_name,
                device=args.device,
            )
        )
        logger.info("HF backend: %s on %s", args.model_name, args.device)

    # 2. Build environment from local JSON
    logger.info("Loading local dataset...")

    async def _setup_env() -> Hermes3DatasetEnv:
        env = Hermes3DatasetEnv.from_local_json(
            json_path=args.json_path,
            limit=args.limit,
            shuffle=True,
            seed=args.seed,
            max_prompt_chars=512,
            val_fraction=0.0,
        )
        await env.setup()
        return env

    env = asyncio.run(_setup_env())
    logger.info("Environment: %d items", len(env.items))

    # 3. Build reward
    reward = CompositeSimilarityReward(
        metrics=[("rouge_l", 0.5), ("bleu", 0.3), ("exact_match", 0.2)]
    )
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
        grad_clip=args.grad_clip,
        bootstrap_sft_rounds=args.bootstrap_sft_rounds,
        bootstrap_sft_samples=args.bootstrap_sft_samples,
        bootstrap_sft_lr=args.bootstrap_sft_lr,
        bootstrap_sft_epochs=args.bootstrap_sft_epochs,
        checkpoint_every=1000,
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

    # Save stats
    stats_path = Path(args.save_stats)
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
