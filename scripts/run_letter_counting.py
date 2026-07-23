#!/usr/bin/env python3
"""Real GRPO training on SmolLM2-360M-Instruct + LoRA + MPS.

    python scripts/run_letter_counting.py

Configurable via env vars:
    DEVICE=mps|cpu           (default: mps)
    MODEL=model_id           (default: HuggingFaceTB/SmolLM2-360M-Instruct)
    LORA_R=rank              (default: 16)
    LORA_ALPHA=alpha         (default: 32.0)
    N_ITERS=iters            (default: 100)
    GROUP_SIZE=G             (default: 4)
    PROMPTS_PER_ITER=P       (default: 4)
    MAX_NEW_TOKENS=N         (default: 64)
    TEMPERATURE=T            (default: 1.0)
    LR=lr                    (default: 1e-4)
    OUTPUT_DIR=path          (default: output/letter_counting)
"""

import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def main():
    from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.letter_counting import (
        LetterCountingConfig,
        LetterCountingEnv,
        LetterCountingReward,
    )
    from hermes_agentic_rl.peft.lora import LoRAConfig, inject_lora
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    # ------------------------------------------------------------------
    # Config from env vars
    # ------------------------------------------------------------------
    device = os.environ.get("DEVICE", "mps")
    model_id = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = float(os.environ.get("LORA_ALPHA", "32.0"))
    n_iters = int(os.environ.get("N_ITERS", "100"))
    group_size = int(os.environ.get("GROUP_SIZE", "4"))
    prompts_per_iter = int(os.environ.get("PROMPTS_PER_ITER", "4"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    temperature = float(os.environ.get("TEMPERATURE", "1.0"))
    lr = float(os.environ.get("LR", "1e-4"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", str(PROJECT / "output/letter_counting")))

    print("=" * 60)
    print("  hermes GRPO + SmolLM2-360M-Instruct + LoRA + Letter Counting")
    print(f"  device={device}  model={model_id}")
    print(f"  LoRA r={lora_r} alpha={lora_alpha}")
    print(f"  iters={n_iters} group={group_size} prompts/iter={prompts_per_iter}")
    print(f"  max_new_tokens={max_new_tokens} temperature={temperature} lr={lr}")
    print(f"  output={output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading model...")
    t0 = time.time()
    backend_cfg = HFBackendConfig(
        model_name_or_path=model_id,
        device=device,
        dtype="float32",
    )
    backend = HFCausalLMBackend(backend_cfg)
    total_params = sum(p.numel() for p in backend.model.parameters())
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Params: {total_params:,}  Vocab: {backend.tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # 2. Inject LoRA (or full fine-tune)
    # ------------------------------------------------------------------
    print("\n[2/5] Configuring trainable parameters...")
    adapter = None
    if lora_r > 0:
        lora_cfg = LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=0.0,
            target_patterns=("c_attn", "c_proj", "c_fc", "q_proj", "v_proj", "o_proj"),
        )
        try:
            adapter = inject_lora(backend.model, lora_cfg)
            print(f"  LoRA modules: {len(adapter.modules)}")
            print(f"  LoRA params: {adapter.num_parameters():,}")
        except ValueError as e:
            print(f"  LoRA injection failed ({e}), falling back to full fine-tune")
            lora_r = 0
    if adapter is None:
        # Full fine-tune
        for p in backend.model.parameters():
            p.requires_grad_(True)
        print("  Full fine-tune mode")
    trainable = sum(p.numel() for p in backend.trainable_parameters())
    print(f"  Trainable: {trainable:,}  Frozen: {total_params - trainable:,}")

    # ------------------------------------------------------------------
    # 3. Build env + reward
    # ------------------------------------------------------------------
    print("\n[3/5] Building env...")
    env_cfg = LetterCountingConfig(
        starting_level=2,
        min_level=1,
        max_level=6,  # cap at 6 for tiny model
        promote_threshold=0.5,
        demote_threshold=0.2,
        window_size=50,
        seed=42,
    )
    env = LetterCountingEnv(env_cfg)
    reward = LetterCountingReward(weight=1.0)
    reward_manager = RewardManager([reward])

    # ------------------------------------------------------------------
    # 4. Build trainer
    # ------------------------------------------------------------------
    print("\n[4/5] Building trainer...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = GRPOTrainerConfig(
        n_iters=n_iters,
        group_size=group_size,
        prompts_per_iter=prompts_per_iter,
        lr=lr,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        clip_eps=0.2,
        kl_coef=0.0,
        entropy_coef=0.01,  # small entropy bonus for exploration
        use_reference=False,
        log_every=5,
        save_every=25,
        output_dir=output_dir,
        seed=42,
        grad_clip=1.0,
        loss_agg="mean_token",
    )
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=tcfg,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[5/5] Training...")
    print("-" * 60)
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    print("-" * 60)
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed / n_iters:.1f}s/iter)")

    # Final summary
    print("\n=== Final Results ===")
    print(f"  Iterations: {n_iters}")
    print(f"  First reward: {stats.iters[0]['mean_reward']:.4f}")
    print(f"  Last reward:  {stats.last_reward():.4f}")
    print(f"  Best reward:  {stats.best_reward():.4f}")
    print(f"  Delta:        {stats.mean_reward_delta():+.4f}")

    # Print reward curve
    print("\n=== Reward Curve ===")
    for rec in stats.iters:
        it = rec["iter"]
        r = rec["mean_reward"]
        loss = rec.get("loss", 0)
        bar = "#" * max(1, int(r * 40))
        print(f"  iter {it:4d} | reward={r:.4f} loss={loss:.4f} | {bar}")

    # Save summary
    import json
    summary = {
        "model": model_id,
        "device": device,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "n_iters": n_iters,
        "group_size": group_size,
        "prompts_per_iter": prompts_per_iter,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "lr": lr,
        "elapsed_seconds": elapsed,
        "first_reward": stats.iters[0]["mean_reward"],
        "last_reward": stats.last_reward(),
        "best_reward": stats.best_reward(),
        "delta": stats.mean_reward_delta(),
        "reward_curve": [
            {"iter": r["iter"], "reward": r["mean_reward"], "loss": r.get("loss", 0)}
            for r in stats.iters
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {output_dir / 'summary.json'}")

    # Save weights
    if adapter is not None:
        adapter_path = output_dir / "lora_adapter.pt"
        adapter.save(adapter_path)
        print(f"LoRA weights saved to {adapter_path}")
    else:
        ckpt_path = output_dir / "policy_full.pt"
        import torch as _torch
        _torch.save(backend.model.state_dict(), ckpt_path)
        print(f"Full model saved to {ckpt_path}")

    return stats


if __name__ == "__main__":
    main()
