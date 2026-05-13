#!/usr/bin/env python3
"""Complete training pipeline on MPS: SFT warmup → GRPO on letter_counting.

    python scripts/train_mps.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
OUTPUT = PROJECT / "output/letter_counting_mps"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = "mps"
MODEL_ID = "openai-community/gpt2"
SFT_SAMPLES = 200
SFT_EPOCHS = 3
SFT_BATCH = 4
SFT_LR = 1e-4
GRPO_ITERS = 50
GRPO_GROUP = 4
GRPO_PROMPTS = 4
GRPO_MAX_TOKENS = 30
GRPO_LR = 5e-5
GRPO_TEMP = 1.0
GRPO_ENTROPY = 0.01


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_ids, answer_ids = self.samples[idx]
        full = prompt_ids + answer_ids
        return torch.tensor(full, dtype=torch.long)


def collate_and_pad(batch):
    max_len = max(t.size(0) for t in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, t in enumerate(batch):
        n = t.size(0)
        padded[i, :n] = t
        mask[i, :n] = True
    return padded[:, :-1], padded[:, 1:], mask[:, 1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from hermes_agentic_rl.backends.hf import HFCausalLMBackend, HFBackendConfig
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.letter_counting import LetterCountingEnv, LetterCountingReward
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    print("=" * 60)
    print("  hermes GRPO + GPT-2 + MPS + Letter Counting")
    print(f"  device={DEVICE}  model={MODEL_ID}")
    print(f"  SFT: samples={SFT_SAMPLES} epochs={SFT_EPOCHS} lr={SFT_LR}")
    print(f"  GRPO: iters={GRPO_ITERS} group={GRPO_GROUP} lr={GRPO_LR}")
    print(f"  output={OUTPUT}")
    print("=" * 60)

    # ---- 1. Load model on MPS ----
    print("\n[1] Loading GPT-2 on MPS...")
    t0 = time.time()
    cfg = HFBackendConfig(model_name_or_path=MODEL_ID, device=DEVICE, dtype="float32")
    backend = HFCausalLMBackend(cfg)
    total = sum(p.numel() for p in backend.model.parameters())
    print(f"  Loaded in {time.time() - t0:.1f}s, params={total:,}")

    # ---- 2. SFT Warmup ----
    print(f"\n[2] SFT warmup ({SFT_SAMPLES} samples × {SFT_EPOCHS} epochs)...")
    env = LetterCountingEnv()
    asyncio.get_event_loop().run_until_complete(env.setup())

    # generate samples
    samples = []
    for _ in range(SFT_SAMPLES):
        item = asyncio.get_event_loop().run_until_complete(env.get_next_item())
        prompt = env.format_prompt(item)
        prompt_ids = backend.tokenizer.encode(prompt)
        correct = item["correct_counts"]
        targets = item["target_letters"]
        if len(targets) == 1:
            ans = f"<answer>{correct[targets[0]]}</answer>"
        else:
            ans = f"<answer>{json.dumps(dict(correct))}</answer>"
        ans_ids = backend.tokenizer.encode(ans, add_eos=True)
        samples.append((prompt_ids, ans_ids))

    dataset = SFTDataset(samples)
    loader = DataLoader(dataset, batch_size=SFT_BATCH, shuffle=True, collate_fn=collate_and_pad)
    optim = torch.optim.AdamW(backend.model.parameters(), lr=SFT_LR)

    backend.model.train()
    for epoch in range(SFT_EPOCHS):
        total_loss = 0.0
        n = 0
        for inp, labels, loss_mask in loader:
            inp, labels, loss_mask = inp.to(DEVICE), labels.to(DEVICE), loss_mask.to(DEVICE)
            optim.zero_grad()
            logits = backend.model(inp).logits
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
            ).reshape(labels.shape)
            loss = (ce * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n += 1
        print(f"  SFT epoch {epoch + 1}/{SFT_EPOCHS} | loss={total_loss / max(n, 1):.4f}")

    # quick test
    backend.model.eval()
    print("  Post-SFT samples:")
    for i in range(3):
        item = asyncio.get_event_loop().run_until_complete(env.get_next_item())
        prompt = env.format_prompt(item)
        ids = backend.tokenizer.encode(prompt)
        gen = backend.generate(ids, max_new_tokens=20, temperature=1.0, seed=i)
        resp = backend.tokenizer.decode(gen.response_ids)
        print(f"    {item['text'][:15]}... → {resp[:50]}")

    # save SFT checkpoint
    sft_path = OUTPUT / "policy_sft.pt"
    torch.save(backend.model.state_dict(), sft_path)
    print(f"  SFT checkpoint saved to {sft_path}")

    # ---- 3. GRPO Training ----
    print(f"\n[3] GRPO training ({GRPO_ITERS} iters)...")
    # fresh env for GRPO
    env2 = LetterCountingEnv()
    reward = LetterCountingReward(weight=1.0)
    rm = RewardManager([reward])

    tcfg = GRPOTrainerConfig(
        n_iters=GRPO_ITERS,
        group_size=GRPO_GROUP,
        prompts_per_iter=GRPO_PROMPTS,
        lr=GRPO_LR,
        max_new_tokens=GRPO_MAX_TOKENS,
        temperature=GRPO_TEMP,
        clip_eps=0.2,
        entropy_coef=GRPO_ENTROPY,
        log_every=5,
        save_every=25,
        output_dir=OUTPUT,
        seed=42,
        grad_clip=1.0,
        loss_agg="mean_token",
    )
    trainer = GRPOTrainer(policy=backend, env=env2, reward_manager=rm, cfg=tcfg)
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0

    # ---- 4. Results ----
    print(f"\n{'=' * 60}")
    print(f"  Training complete in {elapsed:.1f}s ({elapsed / GRPO_ITERS:.1f}s/iter)")
    print(f"  First reward: {stats.iters[0]['mean_reward']:.4f}")
    print(f"  Last reward:  {stats.last_reward():.4f}")
    print(f"  Best reward:  {stats.best_reward():.4f}")
    print(f"  Delta:        {stats.mean_reward_delta():+.4f}")
    print(f"\n  Reward curve:")
    for rec in stats.iters:
        if rec["iter"] % 5 == 0:
            r = rec["mean_reward"]
            bar = "#" * max(1, int(r * 30))
            print(f"    iter {rec['iter']:3d} | {r:.4f} | {bar}")

    # save summary
    summary = {
        "model": MODEL_ID, "device": DEVICE,
        "sft_samples": SFT_SAMPLES, "sft_epochs": SFT_EPOCHS,
        "grpo_iters": GRPO_ITERS, "grpo_group": GRPO_GROUP,
        "grpo_lr": GRPO_LR, "elapsed": elapsed,
        "first_reward": stats.iters[0]["mean_reward"],
        "last_reward": stats.last_reward(),
        "best_reward": stats.best_reward(),
        "delta": stats.mean_reward_delta(),
        "curve": [{"iter": r["iter"], "reward": r["mean_reward"], "loss": r.get("loss", 0)} for r in stats.iters],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(backend.model.state_dict(), OUTPUT / "policy_final.pt")
    print(f"\n  Results saved to {OUTPUT}/")


if __name__ == "__main__":
    main()