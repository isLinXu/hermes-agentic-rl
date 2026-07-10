"""Supervised Fine-Tuning mixin for OnPolicyTrainer.

Extracted from ``on_policy.py`` to isolate SFT-related logic
(interleaved SFT, bootstrap SFT, supervised batch collation).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from hermes_agentic_rl.envs.base_env import SupervisedSample

if TYPE_CHECKING:
    import torch.optim

    from hermes_agentic_rl.backends.base import LLMBackend
    from hermes_agentic_rl.envs.base_env import BaseEnv
    from hermes_agentic_rl.trainers.on_policy_config import OnPolicyTrainerConfig


class SFTMixin:
    """Mixin providing supervised fine-tuning methods.

    Expects the host class to have:
      - ``self.cfg`` (OnPolicyTrainerConfig)
      - ``self.env`` (BaseEnv)
      - ``self.optim`` (Optimizer)
      - ``self.stats`` (TrainStats)
      - ``self.policy`` (LLMBackend)
      - ``self._prompt_encoder`` (PromptStateEncoder)
      - ``self._trainable_params`` (list[torch.nn.Parameter])
      - ``self._minibatch_rng(iter_idx, epoch_idx)`` -> Random
      - ``self._update_ema_rollout()``
      - ``self._policy_max_sequence_length()`` -> int | None
      - ``self._forward_model_logits(inp)`` -> Tensor
    """

    if TYPE_CHECKING:
        cfg: OnPolicyTrainerConfig
        env: BaseEnv
        optim: torch.optim.Optimizer
        stats: Any
        policy: LLMBackend
        _trainable_params: list[Any]
        _prompt_encoder: Any
        _minibatch_rng: Any
        _update_ema_rollout: Any

    def _maybe_run_interleaved_sft(self, iter_idx: int) -> dict[str, Any]:
        every = max(0, int(self.cfg.interleave_sft_every))
        if every <= 0 or iter_idx <= 0 or iter_idx % every != 0:
            return {}
        samples = asyncio.run(
            self._collect_supervised_samples(int(self.cfg.interleave_sft_samples))
        )
        if not samples:
            raise RuntimeError(
                "interleave_sft is enabled, but the active environment produced no "
                "supervised samples. Implement build_supervised_samples(item) on the env "
                "or disable interleave_sft_every."
            )
        return self._run_supervised_updates(
            samples,
            lr=float(self.cfg.interleave_sft_lr),
            epochs=max(1, int(self.cfg.interleave_sft_epochs)),
        )

    def _maybe_run_bootstrap_sft(self) -> dict[str, Any]:
        rounds = max(0, int(self.cfg.bootstrap_sft_rounds))
        if rounds <= 0:
            return {}
        asyncio.run(self.env.setup())

        losses: list[float] = []
        total_samples = 0
        total_steps = 0
        for _ in range(rounds):
            samples = asyncio.run(
                self._collect_supervised_samples(int(self.cfg.bootstrap_sft_samples))
            )
            if not samples:
                raise RuntimeError(
                    "bootstrap_sft is enabled, but the active environment produced no "
                    "supervised samples. Implement build_supervised_samples(item) on the env "
                    "or disable bootstrap_sft_rounds."
                )
            metrics = self._run_supervised_updates(
                samples,
                lr=float(self.cfg.bootstrap_sft_lr),
                epochs=max(1, int(self.cfg.bootstrap_sft_epochs)),
            )
            if "sft_loss" in metrics:
                losses.append(float(metrics["sft_loss"]))
            total_samples += int(metrics.get("n_sft_samples", 0))
            total_steps += int(metrics.get("n_sft_steps", 0))

        return {
            "iter": -1,
            "algo": "sft_bootstrap",
            "mean_reward": 0.0,
            "loss": (sum(losses) / len(losses)) if losses else 0.0,
            "sft_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "n_sft_samples": total_samples,
            "n_sft_steps": total_steps,
            "bootstrap_sft_rounds": rounds,
        }

    async def _collect_supervised_samples(self, n_items: int) -> list[SupervisedSample]:
        out: list[SupervisedSample] = []
        for _ in range(max(1, n_items)):
            item = await self.env.get_next_item()
            out.extend(self.env.build_supervised_samples(item))
        return [
            sample
            for sample in out
            if str(sample.instruction).strip() and str(sample.response).strip()
        ]

    def _run_supervised_updates(
        self,
        samples: list[SupervisedSample],
        *,
        lr: float,
        epochs: int,
    ) -> dict[str, Any]:
        batch_size = max(1, min(int(self.cfg.interleave_sft_batch_size), len(samples)))
        prev_lrs = [float(group["lr"]) for group in self.optim.param_groups]
        for group in self.optim.param_groups:
            group["lr"] = float(lr)

        losses: list[float] = []
        n_steps = 0
        try:
            for epoch_idx in range(epochs):
                ordered = list(samples)
                rng_epoch = self._minibatch_rng(
                    iter_idx=len(self.stats.iters) + 1,
                    epoch_idx=epoch_idx + 1,
                )
                rng_epoch.shuffle(ordered)
                for start in range(0, len(ordered), batch_size):
                    batch = ordered[start : start + batch_size]
                    if not batch:
                        continue
                    inp, labels, loss_mask = self._collate_supervised_batch(batch)
                    self.optim.zero_grad()
                    logits = self._forward_model_logits(inp)
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction="none",
                    ).reshape(labels.shape)
                    loss = (ce * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
                    loss.backward()
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._trainable_params,
                            max_norm=self.cfg.grad_clip,
                        )
                    self.optim.step()
                    losses.append(float(loss.detach().item()))
                    n_steps += 1
        finally:
            for group, lr in zip(self.optim.param_groups, prev_lrs, strict=False):
                group["lr"] = lr

        return {
            "sft_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "n_sft_samples": len(samples),
            "n_sft_steps": n_steps,
        }

    def _collate_supervised_batch(
        self, batch: list[SupervisedSample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows: list[tuple[list[int], int]] = []
        max_len = self._policy_max_sequence_length()
        for sample in batch:
            obs = self._prompt_encoder.encode({"instruction": sample.instruction})
            prompt_ids = list(obs.prompt_ids)
            if sample.prompt_suffix:
                prompt_ids.extend(self.policy.tokenizer.encode(sample.prompt_suffix))
            response_ids = self.policy.tokenizer.encode(sample.response, add_eos=True)
            full = prompt_ids + response_ids
            if max_len is not None and len(full) > max_len:
                drop = len(full) - max_len
                full = full[drop:]
                prompt_ids = prompt_ids[drop:] if drop < len(prompt_ids) else []
            rows.append((full, len(prompt_ids)))

        max_len = max(len(full_ids) for full_ids, _prompt_len in rows)
        device = self._trainable_params[0].device
        pad_id = int(getattr(self.policy.tokenizer, "pad_id", 0))
        inp = torch.full((len(rows), max_len - 1), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(rows), max_len - 1), pad_id, dtype=torch.long, device=device)
        loss_mask = torch.zeros((len(rows), max_len - 1), dtype=torch.bool, device=device)

        for row_idx, (full_ids, prompt_len) in enumerate(rows):
            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]
            n = len(input_ids)
            if n <= 0:
                continue
            inp[row_idx, :n] = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels[row_idx, :n] = torch.tensor(target_ids, dtype=torch.long, device=device)
            start = max(0, prompt_len - 1)
            loss_mask[row_idx, start:n] = True
        return inp, labels, loss_mask

    def _policy_max_sequence_length(self) -> int | None:
        cfg = getattr(self.policy, "cfg", None)
        for source in (cfg, getattr(self.policy, "model", None)):
            if source is None:
                continue
            for attr in ("max_len", "max_new_tokens", "n_positions", "max_position_embeddings"):
                max_len = getattr(source, attr, None)
                if isinstance(max_len, int) and max_len > 0:
                    return max_len
            # Check model.config for HF models
            model_config = getattr(source, "config", None)
            if model_config is not None:
                for attr in ("n_positions", "max_position_embeddings", "max_length"):
                    max_len = getattr(model_config, attr, None)
                    if isinstance(max_len, int) and max_len > 0:
                        return max_len
        return None
        cfg = getattr(self.policy, "cfg", None)
        for source in (cfg, getattr(self.policy, "model", None)):
            if source is None:
                continue
            for attr in ("max_len", "max_new_tokens", "n_positions", "max_position_embeddings"):
                max_len = getattr(source, attr, None)
                if isinstance(max_len, int) and max_len > 0:
                    return max_len
        return None

    def _forward_model_logits(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.policy.model(inp)  # type: ignore[attr-defined]
        return out.logits if hasattr(out, "logits") else out
