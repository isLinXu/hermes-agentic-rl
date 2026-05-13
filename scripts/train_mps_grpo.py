#!/usr/bin/env python3
"""GRPO training on MPS, starting from CPU-trained SFT checkpoint."""

import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
OUTPUT = PROJECT / "output/letter_counting_mps"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = "mps"
MODEL_ID = "openai-community/gpt2"
SFT_CKPT = PROJECT / "output/letter_counting/policy_sft.pt"
GRPO_ITERS = 100
GRPO_GROUP = 4
GRPO_PROMPTS = 4
GRPO_MAX_TOKENS = 20
GRPO_LR = 2e-5
GRPO_TEMP = 0.8
GRPO_ENTROPY = 0.05


def main():
    from hermes_agentic_rl.backends.hf import HFCausalLMBackend, HFBackendConfig
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.letter_counting import LetterCountingEnv, LetterCountingReward
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    print("=" * 60)
    print("  hermes GRPO + GPT-2 (SFT) + MPS + Letter Counting")
    print(f"  SFT checkpoint: {SFT_CKPT}")
    print(f"  iters={GRPO_ITERS} group={GRPO_GROUP} prompts/iter={GRPO_PROMPTS}")
    print(f"  max_tokens={GRPO_MAX_TOKENS} lr={GRPO_LR} temp={GRPO_TEMP}")
    print("=" * 60)

    # Load model + SFT weights
    print("\n[1] Loading GPT-2 + SFT weights on MPS...")
    t0 = time.time()
    cfg = HFBackendConfig(model_name_or_path=MODEL_ID, device=DEVICE, dtype="float32")
    backend = HFCausalLMBackend(cfg)
    backend.model.load_state_dict(torch.load(SFT_CKPT, map_location=DEVICE))
    total = sum(p.numel() for p in backend.model.parameters())
    print(f"  Loaded in {time.time() - t0:.1f}s, params={total:,}")

    # Quick generate test
    prompt = backend.tokenizer.encode("How many e in elephant?")
    gen = backend.generate(prompt, max_new_tokens=20, temperature=1.0, seed=42)
    print(f"  Sample generate: {backend.tokenizer.decode(gen.response_ids)[:60]}")

    # Build env + reward
    print("\n[2] Building env...")
    env = LetterCountingEnv()
    reward = LetterCountingReward(weight=1.0)
    rm = RewardManager([reward])

    # Train
    print(f"\n[3] GRPO training ({GRPO_ITERS} iters)...")
    tcfg = GRPOTrainerConfig(
        n_iters=GRPO_ITERS,
        group_size=GRPO_GROUP,
        prompts_per_iter=GRPO_PROMPTS,
        lr=GRPO_LR,
        max_new_tokens=GRPO_MAX_TOKENS,
        temperature=GRPO_TEMP,
        clip_eps=0.2,
        kl_coef=0.02,
        entropy_coef=GRPO_ENTROPY,
        use_reference=True,
        log_every=5,
        save_every=25,
        output_dir=OUTPUT,
        seed=42,
        grad_clip=1.0,
        loss_agg="mean_token",
    )
    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=tcfg)
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0

    # Results
    print(f"\n{'=' * 60}")
    print(f"  Complete in {elapsed:.1f}s ({elapsed / GRPO_ITERS:.1f}s/iter)")
    print(f"  First: {stats.iters[0]['mean_reward']:.4f}  Last: {stats.last_reward():.4f}")
    print(f"  Best: {stats.best_reward():.4f}  Delta: {stats.mean_reward_delta():+.4f}")

    print(f"\n  Reward curve:")
    for rec in stats.iters:
        if rec["iter"] % 5 == 0:
            r = rec["mean_reward"]
            bar = "#" * max(1, int(r * 30))
            print(f"    iter {rec['iter']:3d} | {r:.4f} | {bar}")

    summary = {
        "model": MODEL_ID, "device": DEVICE, "sft_ckpt": str(SFT_CKPT),
        "iters": GRPO_ITERS, "group": GRPO_GROUP, "lr": GRPO_LR,
        "elapsed": elapsed,
        "first": stats.iters[0]["mean_reward"],
        "last": stats.last_reward(),
        "best": stats.best_reward(),
        "delta": stats.mean_reward_delta(),
        "curve": [{"iter": r["iter"], "reward": r["mean_reward"]} for r in stats.iters],
    }
    (OUTPUT / "summary_grpo_sft.json").write_text(json.dumps(summary, indent=2))
    torch.save(backend.model.state_dict(), OUTPUT / "policy_grpo_final.pt")
    print(f"\n  Saved to {OUTPUT}/")


if __name__ == "__main__":
    main()