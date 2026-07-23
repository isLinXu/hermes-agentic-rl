"""Standalone training script with CSV metrics logging.

Logs per-iteration metrics to a CSV file for post-hoc visualization.
Usage::

    python scripts/train_hermes3_real_with_csv.py \
        --backend tiny --limit 100 --n-iters 20 \
        --metrics-csv ./outputs/metrics.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend  # noqa: E402
from hermes_agentic_rl.core.reward_manager import RewardManager  # noqa: E402
from hermes_agentic_rl.envs.hermes3_dataset_env import Hermes3DatasetEnv  # noqa: E402
from hermes_agentic_rl.rewards.similarity_reward import CompositeSimilarityReward  # noqa: E402
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig  # noqa: E402

logger = logging.getLogger("train_hermes3_real")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train on Hermes-3-Dataset with CSV logging"
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
        choices=["rouge_l", "rouge_1", "rouge_2", "bleu",
                 "exact_match", "token_overlap", "composite"],
        default="composite",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--metrics-csv", default="./outputs/metrics.csv")
    p.add_argument("--output-dir", default="./outputs/hermes3_real")
    return p.parse_args()


class MetricsLogger:
    """Append per-iteration metrics to a CSV file."""

    FIELDNAMES = [
        "iter", "algo", "mean_reward", "loss", "policy_loss",
        "mean_advantage", "kl", "clip_frac", "n_updated",
        "entropy", "n_records", "n_optimizer_steps", "update_epochs",
        "n_minibatches", "minibatch_size", "approx_kl",
        "score_temperature", "grad_norm", "param_norm", "lr",
        "grad_clip_triggered", "reward_min", "reward_max",
        "reward_std", "n_groups", "finished_naturally_rate",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._initialized = False

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if not exists:
            self._writer.writeheader()
        self._initialized = True

    def log(self, metrics: dict[str, Any]) -> None:
        self._ensure_open()
        row = {k: metrics.get(k, "") for k in self.FIELDNAMES}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Hermes-Agentic-RL: Scaled Training with CSV Logging")
    logger.info("=" * 60)
    logger.info("Dataset: %s (split=%s)", args.dataset_name, args.dataset_split)
    logger.info("Backend: %s", args.backend)
    logger.info("Iters: %d, Group: %d", args.n_iters, args.group_size)
    logger.info("Limit: %s", args.limit if args.limit else "full")
    logger.info("Metrics CSV: %s", args.metrics_csv)

    metrics_logger = MetricsLogger(args.metrics_csv)

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
        from hermes_agentic_rl.backends.hf_backend import HFBackend, HFBackendConfig
        backend = HFBackend(
            HFBackendConfig(
                model_name=args.model_name,
                device=args.device,
                max_new_tokens=args.max_seq_len,
            )
        )
        logger.info("HF: %s on %s", args.model_name, args.device)

    # 2. Build environment
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

    # Monkey-patch trainer._log_iter to also write CSV
    original_log_iter = trainer._log_iter

    def _patched_log_iter(iter_idx: int, metrics: dict) -> None:
        original_log_iter(iter_idx, metrics)
        metrics_logger.log(metrics)

    trainer._log_iter = _patched_log_iter  # type: ignore[method-assign]

    logger.info("Starting training...")
    try:
        stats = trainer.train()
    finally:
        metrics_logger.close()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Stats: %s", stats)
    logger.info("Metrics saved to: %s", args.metrics_csv)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
