#!/usr/bin/env python3
"""Quick SFT warmup: teach gpt2 to output <answer>N</answer> format.

Generates synthetic samples from the letter_counting env, then runs a few
steps of next-token prediction (standard LM loss) so the model learns to
produce the correct answer format.

After this, GRPO training should start getting non-zero reward.
"""

import asyncio
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


class SFTDataset(Dataset):
    def __init__(self, samples: list[tuple[list[int], list[int]]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_ids, answer_ids = self.samples[idx]
        full = prompt_ids + answer_ids
        return torch.tensor(full, dtype=torch.long)


def collate_and_pad(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(t.size(0) for t in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, t in enumerate(batch):
        n = t.size(0)
        padded[i, :n] = t
        mask[i, :n] = True
    # Shift: input = tokens[:-1], target = tokens[1:]
    input_ids = padded[:, :-1]
    labels = padded[:, 1:]
    loss_mask = mask[:, 1:]
    return input_ids, labels, loss_mask


def generate_samples(env, backend, n: int = 200) -> list[tuple[list[int], list[int]]]:
    """Generate (prompt_ids, answer_ids) pairs."""
    import json

    samples = []
    for _ in range(n):
        item = asyncio.get_event_loop().run_until_complete(env.get_next_item())
        prompt = env.format_prompt(item)
        prompt_ids = backend.tokenizer.encode(prompt)

        correct = item["correct_counts"]
        targets = item["target_letters"]
        if len(targets) == 1:
            answer_text = f"<answer>{correct[targets[0]]}</answer>"
        else:
            answer_text = f"<answer>{json.dumps(dict(correct))}</answer>"
        answer_ids = backend.tokenizer.encode(answer_text, add_eos=True)
        samples.append((prompt_ids, answer_ids))
    return samples


def main():
    from hermes_agentic_rl.backends.hf import HFCausalLMBackend, HFBackendConfig
    from hermes_agentic_rl.envs.letter_counting import LetterCountingEnv

    device = "cpu"
    model_id = "openai-community/gpt2"
    n_samples = 200
    n_epochs = 3
    batch_size = 4
    lr = 1e-4

    print("=" * 60)
    print("  SFT Warmup: teach gpt2 to output <answer> format")
    print(f"  model={model_id}  samples={n_samples}  epochs={n_epochs}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    cfg = HFBackendConfig(model_name_or_path=model_id, device=device, dtype="float32")
    backend = HFCausalLMBackend(cfg)
    for p in backend.model.parameters():
        p.requires_grad_(True)

    # Generate samples
    print(f"\n[2/3] Generating {n_samples} SFT samples...")
    env = LetterCountingEnv()
    asyncio.get_event_loop().run_until_complete(env.setup())
    samples = generate_samples(env, backend, n=n_samples)
    print(f"  Generated {len(samples)} samples")
    # Show a few
    for i in range(3):
        p_ids, a_ids = samples[i]
        print(f"  Sample {i}: prompt_len={len(p_ids)}, answer_len={len(a_ids)}")
        print(f"    answer: {backend.tokenizer.decode(a_ids)}")

    # Train
    print(f"\n[3/3] SFT training ({n_epochs} epochs, batch={batch_size}, lr={lr})...")
    dataset = SFTDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_and_pad)
    optim = torch.optim.AdamW(backend.model.parameters(), lr=lr)

    backend.model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for input_ids, labels, loss_mask in loader:
            optim.zero_grad()
            logits = backend.model(input_ids).logits
            # cross-entropy on answer tokens only
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            )
            ce = ce.reshape(labels.shape)
            masked_ce = (ce * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
            masked_ce.backward()
            optim.step()
            total_loss += masked_ce.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch + 1}/{n_epochs} | loss={avg_loss:.4f}")

    # Test generation after SFT
    print("\n--- Post-SFT generation test ---")
    backend.model.eval()
    for i in range(5):
        item = asyncio.get_event_loop().run_until_complete(env.get_next_item())
        prompt = env.format_prompt(item)
        ids = backend.tokenizer.encode(prompt)
        gen = backend.generate(ids, max_new_tokens=20, temperature=1.0, seed=i)
        response = backend.tokenizer.decode(gen.response_ids)
        correct = item["correct_counts"]
        print(f"  Q: {item['text'][:20]}... → {response[:60]}  (expected: {correct})")

    # Save
    out = PROJECT / "output/letter_counting"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "policy_sft.pt"
    torch.save(backend.model.state_dict(), ckpt)
    print(f"\nSFT model saved to {ckpt}")


if __name__ == "__main__":
    main()