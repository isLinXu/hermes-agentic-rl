#!/usr/bin/env python3
"""v0.5 End-to-end RL training with all enhancements.

Features:
  - HF backend (GPT-2 on MPS/CPU)
  - Letter counting env with curriculum
  - Interleaved SFT + GRPO (anti-collapse)
  - Per-token advantage (REINFORCE++)
  - Batch generate (KV-cache)
  - Checkpoint save/restore
  - Live dashboard

Usage::

    # Quick smoke test (10 iters, CPU)
    python scripts/train_v05.py --iters 10 --device cpu --smoke

    # Full MPS training
    python scripts/train_v05.py --iters 200 --device mps

    # Resume from checkpoint
    python scripts/train_v05.py --resume iter_00050

    # Code-fix environment
    python scripts/train_v05.py --env code_fix --iters 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def parse_args():
    p = argparse.ArgumentParser(description="v0.5 RL Training")
    p.add_argument("--iters", type=int, default=100, help="Training iterations")
    p.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    p.add_argument("--model", type=str, default="gpt2", help="HF model name")
    p.add_argument(
        "--env", type=str, default="letter_counting",
        choices=["letter_counting", "code_fix"]
    )
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--batch-size", type=int, default=2, help="Prompts per iter")
    p.add_argument("--group-size", type=int, default=4, help="Rollouts per prompt (GRPO)")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Max generation tokens")
    p.add_argument("--kl-coef", type=float, default=0.04, help="KL penalty coefficient")
    p.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy bonus")
    p.add_argument(
        "--interleave-sft", type=int, default=10,
        help="Interleave SFT every N iters (0=off)"
    )
    p.add_argument("--per-token-adv", action="store_true", default=True, help="Per-token advantage")
    p.add_argument(
        "--no-per-token-adv", action="store_false", dest="per_token_adv",
        help="Disable per-token adv"
    )
    p.add_argument(
        "--advantage-norm", type=str, default="whiten",
        choices=["group", "batch", "whiten"]
    )
    p.add_argument(
        "--batch-generate", action="store_true", default=True,
        help="Use KV-cache batch generate"
    )
    p.add_argument(
        "--no-batch-generate", action="store_false", dest="batch_generate",
        help="Disable batch generate"
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint iteration")
    p.add_argument("--dashboard", action="store_true", default=False, help="Enable live dashboard")
    p.add_argument("--dashboard-port", type=int, default=8765)
    p.add_argument("--smoke", action="store_true", default=False, help="Smoke test (10 iters, CPU)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


async def build_env(env_type: str, device: str, seed: int):
    """Build the training environment."""
    from hermes_agentic_rl.core.reward_manager import RewardManager

    if env_type == "letter_counting":
        from hermes_agentic_rl.envs.letter_counting import (
            LetterCountingConfig,
            LetterCountingEnv,
            LetterCountingReward,
        )

        env = LetterCountingEnv(
            LetterCountingConfig(max_level=10, seed=seed)
        )
        rm = RewardManager([LetterCountingReward(weight=1.0)])
        return env, rm

    elif env_type == "code_fix":
        from hermes_agentic_rl.envs.code_fix import CodeFixEnv, CodeFixReward

        env = CodeFixEnv(seed=seed, n_samples=100, max_level=2)
        rm = RewardManager([CodeFixReward(weight=1.0)])
        return env, rm

    raise ValueError(f"Unknown env: {env_type}")


def build_backend(model_name: str, device: str, seed: int):
    """Build HF backend."""
    from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend

    print(f"[init] loading {model_name} on {device}...")
    t0 = time.time()

    dtype = "float32"
    if device == "mps":
        dtype = "float32"  # MPS is float32-only for most ops

    backend = HFCausalLMBackend(
        HFBackendConfig(
            model_name_or_path=model_name,
            device=device,
            dtype=dtype,
            with_value_head=False,
            trust_remote_code=False,
        )
    )
    print(f"[init] loaded in {time.time() - t0:.1f}s")

    # Set seed
    import torch
    torch.manual_seed(seed)

    return backend


def build_trainer(
    backend,
    env,
    reward_manager,
    args,
    output_dir: Path | None,
):
    """Build GRPO trainer with v0.5 enhancements."""
    from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    cfg = GRPOTrainerConfig(
        n_iters=args.iters,
        group_size=args.group_size,
        prompts_per_iter=args.batch_size,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        clip_eps=0.2,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        log_every=1,
        save_every=50,
        output_dir=output_dir,
        seed=args.seed,
        per_token_advantage=args.per_token_adv,
        advantage_norm=args.advantage_norm,
        interleave_sft_every=args.interleave_sft if not args.smoke else 0,
        interleave_sft_samples=32,
        interleave_sft_lr=args.lr * 2,
        batch_generate=args.batch_generate,
        grad_clip=1.0,
    )

    GRPO(
        GRPOConfig(
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
            entropy_coef=cfg.entropy_coef,
            per_token_advantage=cfg.per_token_advantage,
            advantage_norm=cfg.advantage_norm,
            loss_agg="mean_token",
        )
    )

    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=cfg,
    )

    return trainer, cfg


async def main():
    args = parse_args()

    # Smoke test override
    if args.smoke:
        args.iters = 10
        args.device = "cpu"
        args.batch_size = 1
        args.group_size = 2
        args.interleave_sft = 0

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _project_root / "output" / f"v05_{args.env}_{args.device}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] output_dir={output_dir}")

    # Build components
    env, reward_manager = await build_env(args.env, args.device, args.seed)
    backend = build_backend(args.model, args.device, args.seed)
    trainer, _cfg = build_trainer(backend, env, reward_manager, args, output_dir)

    # Dashboard
    dashboard = None
    if args.dashboard:
        from hermes_agentic_rl.monitor.dashboard import LiveDashboard

        dashboard = LiveDashboard(host="127.0.0.1", port=args.dashboard_port)
        url = dashboard.start()
        print(f"[dashboard] live at {url}")

    # Resume
    start_iter = 0
    if args.resume:
        from hermes_agentic_rl.trainers.checkpoint import (
            CheckpointManager,
            restore_checkpoint_to_trainer,
        )

        ckpt = CheckpointManager(output_dir / "checkpoints")
        if args.resume == "latest":
            state = ckpt.load_latest()
        else:
            try:
                it = int(args.resume.replace("iter_", ""))
                state = ckpt.load(it)
            except ValueError:
                state = ckpt.load_latest()

        if state is not None:
            start_iter = restore_checkpoint_to_trainer(trainer, state)
            print(f"[resume] restored from iter {start_iter}")
        else:
            print("[resume] no checkpoint found, starting fresh")

    # Training (train() internally uses asyncio.run, so don't wrap in another)
    print(
        f"[train] algo=grpo model={args.model} device={args.device} "
        f"iters={args.iters} group={args.group_size} batch={args.batch_size} "
        f"lr={args.lr} temp={args.temperature} "
        f"per_tok_adv={args.per_token_adv} adv_norm={args.advantage_norm} "
        f"interleave_sft={args.interleave_sft} batch_gen={args.batch_generate}"
    )

    t_start = time.time()
    try:
        stats = trainer.train()
    finally:
        if dashboard:
            dashboard.stop()

    elapsed = time.time() - t_start

    # Save summary
    summary = {
        "version": "0.5.0",
        "model": args.model,
        "device": args.device,
        "env": args.env,
        "iters": args.iters,
        "elapsed_s": elapsed,
        "iters_per_s": args.iters / elapsed if elapsed > 0 else 0,
        "last_mean_reward": stats.last_reward() if hasattr(stats, "last_reward") else None,
        "best_mean_reward": stats.best_reward() if hasattr(stats, "best_reward") else None,
        "reward_history": stats.iters if hasattr(stats, "iters") else [],
    }
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n{'='*60}")
    print(f"[done] elapsed={elapsed:.1f}s ({elapsed/args.iters:.1f}s/iter)")
    if hasattr(stats, "best_reward"):
        print(f"[done] best_reward={stats.best_reward():.4f}")
    if hasattr(stats, "last_reward"):
        print(f"[done] last_reward={stats.last_reward():.4f}")
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    # Don't use asyncio.run here — train() already manages its own event loop
    import asyncio as _asyncio

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
