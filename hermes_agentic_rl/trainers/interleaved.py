"""Interleaved SFT + GRPO trainer — prevents format collapse.

Core insight from our GPT-2 experiment: GRPO alone drifts the model away
from the correct answer format (<answer>...</answer>), causing reward collapse.
Solution: every N GRPO steps, run one SFT mini-batch on correct-format
samples to "remind" the model of the expected output structure.

This is a thin wrapper: it takes an existing GRPOTrainer and periodically
injects SFT steps using a small buffer of ground-truth samples.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(slots=True)
class InterleaveConfig:
    """Configuration for interleaved SFT + GRPO training.

    sft_every: run SFT step every N GRPO iterations (0 = never).
    sft_samples: number of fresh ground-truth samples to generate per SFT step.
    sft_epochs: SFT epochs per interleave (usually 1 for online setting).
    sft_batch_size: SFT mini-batch size.
    sft_lr: SFT learning rate (can be higher than GRPO lr since it's corrective).
    """

    sft_every: int = 5
    sft_samples: int = 32
    sft_epochs: int = 1
    sft_batch_size: int = 4
    sft_lr: float = 1e-4


class SFTBuffer(Dataset):
    """Holds (prompt_ids, answer_ids) pairs for format-reminder SFT."""

    def __init__(self):
        self.samples: list[tuple[list[int], list[int]]] = []

    def add(self, prompt_ids: list[int], answer_ids: list[int]):
        self.samples.append((prompt_ids, answer_ids))

    def clear(self):
        self.samples.clear()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_ids, answer_ids = self.samples[idx]
        full = prompt_ids + answer_ids
        return torch.tensor(full, dtype=torch.long)


def _collate_sft(batch):
    max_len = max(t.size(0) for t in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, t in enumerate(batch):
        n = t.size(0)
        padded[i, :n] = t
        mask[i, :n] = True
    return padded[:, :-1], padded[:, 1:], mask[:, 1:]


class InterleavedGRPOTrainer:
    """Wraps a GRPOTrainer and interleaves SFT format-reminder steps.

    Usage::

        trainer = InterleavedGRPOTrainer(
            grpo_trainer=grpo,
            env=env,
            backend=backend,
            cfg=InterleaveConfig(sft_every=5),
        )
        stats = trainer.train()
    """

    def __init__(
        self,
        grpo_trainer: Any,  # GRPOTrainer
        env: Any,  # BaseEnv
        backend: Any,  # LLMBackend
        cfg: InterleaveConfig | None = None,
        device: str = "cpu",
    ):
        self._grpo = grpo_trainer
        self._env = env
        self._backend = backend
        self.cfg = cfg or InterleaveConfig()
        self.device = device
        self._sft_optim: torch.optim.Optimizer | None = None
        self._buffer = SFTBuffer()
        self._all_stats: list[dict[str, Any]] = []

    def _generate_sft_samples(self, n: int) -> None:
        """Generate fresh ground-truth format samples from the env."""
        self._buffer.clear()
        for _ in range(n):
            item = asyncio.get_event_loop().run_until_complete(self._env.get_next_item())
            prompt = self._env.format_prompt(item)
            prompt_ids = self._backend.tokenizer.encode(prompt)
            correct = item["correct_counts"]
            targets = item["target_letters"]
            if len(targets) == 1:
                answer = f"<answer>{correct[targets[0]]}</answer>"
            else:
                answer = f"<answer>{json.dumps(dict(correct))}</answer>"
            answer_ids = self._backend.tokenizer.encode(answer, add_eos=True)
            self._buffer.add(prompt_ids, answer_ids)

    def _sft_step(self) -> float:
        """Run one SFT epoch on the buffer."""
        if len(self._buffer) == 0:
            return 0.0

        if self._sft_optim is None:
            self._sft_optim = torch.optim.AdamW(
                self._backend.trainable_parameters(), lr=self.cfg.sft_lr
            )

        loader = DataLoader(
            self._buffer,
            batch_size=self.cfg.sft_batch_size,
            shuffle=True,
            collate_fn=_collate_sft,
        )

        self._backend.model.train()
        total_loss = 0.0
        n = 0
        for _epoch in range(self.cfg.sft_epochs):
            for inp, labels, loss_mask in loader:
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                loss_mask = loss_mask.to(self.device)
                self._sft_optim.zero_grad()
                logits = self._backend.model(inp).logits
                ce = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape(labels.shape)
                loss = (ce * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
                loss.backward()
                self._sft_optim.step()
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    def train(self) -> Any:
        """Run interleaved training.

        Returns the accumulated stats (same format as GRPOTrainer.train()).
        """
        from hermes_agentic_rl.trainers.on_policy import TrainStats

        stats = TrainStats()

        for it in range(self._grpo.cfg.n_iters):
            # --- GRPO step ---
            grpo_stats = asyncio.run(self._grpo._one_iter(it))
            record = {"iter": it, "algo": "grpo", **grpo_stats.as_dict()}
            stats.add(record)
            self._all_stats.append(record)

            if self._grpo.cfg.log_every and (it % self._grpo.cfg.log_every == 0):
                self._grpo.logger(record)

            # --- Interleaved SFT ---
            if self.cfg.sft_every > 0 and it > 0 and it % self.cfg.sft_every == 0:
                self._generate_sft_samples(self.cfg.sft_samples)
                sft_loss = self._sft_step()
                record_sft = {
                    "iter": it,
                    "algo": "sft_interleave",
                    "loss": sft_loss,
                    "mean_reward": record.get("mean_reward", 0),
                }
                stats.add(record_sft)
                self._all_stats.append(record_sft)
                if self._grpo.cfg.log_every:
                    print(f"[train] iter={it} algo=sft_interleave sft_loss={sft_loss:.4f}")

            # --- Checkpoint ---
            if (
                self._grpo.cfg.save_every
                and self._grpo.cfg.output_dir is not None
                and self._grpo.cfg.save_every > 0
                and it > 0
                and it % self._grpo.cfg.save_every == 0
            ):
                self._grpo._save_checkpoint(it)

        return stats


def _extract_grpo_reward_records(stats: Any) -> list[dict[str, Any]]:
    """Filter stats to only GRPO records (for reward curve plotting)."""
    return [s for s in stats.iters if s.get("algo") == "grpo"]
