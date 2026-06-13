"""PRM Training Pipeline — Online co-training of ProcessRewardModel.

Integrates PRM self-improvement into the on-policy RL loop:

  1. ``PRMPipelineConfig`` — configuration for auto-generation of step labels.
  2. ``PRMSampleGenerator`` — converts RolloutBatch → PRMStepSamples using
     ORM outcome pseudo-labels (positive rollouts supply positive steps,
     negative rollouts supply negative steps; intermediate steps get
     discounted labels).
  3. ``PRMPipeline`` — wrapper that calls ``PRMTrainer.train()`` every
     ``train_every`` policy iterations, using freshly generated samples and
     an optional replay buffer of past samples.

Usage in OnPolicyTrainer subclass::

    from hermes_agentic_rl.trainers.prm_pipeline import PRMPipeline, PRMPipelineConfig
    from hermes_agentic_rl.rewards.prm import ProcessRewardModel, PRMTrainer

    prm_pipeline = PRMPipeline(
        prm=prm_model,
        cfg=PRMPipelineConfig(train_every=5, samples_per_iter=64),
    )
    # In the training loop, after collecting a batch:
    prm_metrics = prm_pipeline.step(it, batch, tokenizer=policy.tokenizer)

Design notes:
  - ORM pseudo-labels: outcome reward > 0 → all steps labeled 1.0;
    outcome reward ≤ 0 → label is ``gamma^(n_steps - 1 - t)`` (decays for
    early steps that may still be correct).
  - Credit decay (``step_discount`` γ): intermediate steps receive
    ``reward * gamma^(T-1-t)`` so only the final step of a positive
    rollout is strongly labeled; early steps are softer.
  - Replay: a simple FIFO buffer of capacity ``replay_capacity`` retains
    samples from previous iters so rare positive examples are not wasted.
  - Frequency: PRM is trained every ``train_every`` RL iters to amortise
    the backward cost.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.rewards.prm import (
    PRMConfig,
    PRMStepSample,
    PRMTrainer,
    ProcessRewardModel,
    split_steps,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PRMPipelineConfig:
    """Configuration for the online PRM co-training pipeline."""

    # How often (in RL iterations) to run a PRM training round.
    # 0 = disabled.
    train_every: int = 5

    # Max PRM training samples generated per RL iter (before replay mixing).
    samples_per_iter: int = 64

    # Discount applied to intermediate steps for pseudo-labels.
    # step_discount^(T-1-t) where T = n_steps, t = step index.
    step_discount: float = 0.95

    # Reward threshold: rollouts with final_reward > threshold are labeled
    # positive; ≤ threshold are labeled negative.
    positive_threshold: float = 0.5

    # Replay buffer: retain this many past PRMStepSamples across iters.
    # 0 = no replay (pure on-policy PRM training).
    replay_capacity: int = 512

    # Fraction of each PRM training batch drawn from the replay buffer.
    replay_ratio: float = 0.25

    # PRM model training config.
    prm_cfg: PRMConfig = field(default_factory=PRMConfig)

    # Step separator string (must match PRMComponent's step_sep).
    step_sep: str = "\n\n"

    # When True, skip records whose response decodes to empty string.
    skip_empty_responses: bool = True

    # Minimum number of steps required to generate a sample.
    min_steps: int = 1


# ---------------------------------------------------------------------------
# Sample generator
# ---------------------------------------------------------------------------


class PRMSampleGenerator:
    """Convert a RolloutBatch into PRMStepSample training examples.

    ORM pseudo-label strategy
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Positive rollout (reward > threshold):
      - Step t gets label = ``step_discount^(T-1-t)``.  Final step gets 1.0,
        earlier steps get softer positives (they are likely correct but we
        are less certain).

    Negative rollout (reward ≤ threshold):
      - All steps get label = 0.0 (bad reasoning chain).

    This avoids the need for step-level human annotations while providing
    a dense training signal that matches the ORM outcome.
    """

    def __init__(
        self,
        cfg: PRMPipelineConfig,
        tokenizer: Any,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer

    def generate(
        self,
        batch: RolloutBatch,
        *,
        max_samples: int | None = None,
    ) -> list[PRMStepSample]:
        """Generate PRMStepSamples from a batch of rollout records."""
        cfg = self.cfg
        max_s = max_samples or cfg.samples_per_iter
        samples: list[PRMStepSample] = []
        records = list(batch.records)
        random.shuffle(records)

        for rec in records:
            if len(samples) >= max_s:
                break
            new_samples = self._record_to_samples(rec)
            samples.extend(new_samples)
            if len(samples) > max_s:
                samples = samples[:max_s]
                break

        return samples

    def _record_to_samples(self, rec: RolloutRecord) -> list[PRMStepSample]:
        cfg = self.cfg
        prompt_ids = list(rec.prompt_ids)
        response_ids = list(rec.response_ids)

        if not response_ids:
            return []

        # Decode response text for step splitting.
        try:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        except Exception:
            return []

        if cfg.skip_empty_responses and not response_text.strip():
            return []

        steps = split_steps(response_text, sep=cfg.step_sep)
        if len(steps) < cfg.min_steps:
            return []

        is_positive = float(rec.reward) > cfg.positive_threshold
        T = len(steps)
        samples: list[PRMStepSample] = []

        for t, _step_text in enumerate(steps):
            # Build prefix up to and including step t.
            prefix = cfg.step_sep.join(steps[: t + 1])
            try:
                prefix_ids = self.tokenizer.encode(prefix)
            except Exception:
                continue

            label = float(cfg.step_discount ** (T - 1 - t)) if is_positive else 0.0

            samples.append(
                PRMStepSample(
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix_ids,
                    label=label,
                )
            )

        return samples


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PRMPipeline:
    """Online co-training loop for ProcessRewardModel.

    Designed to be called from within the RL training loop::

        prm_pipeline = PRMPipeline(prm=model, cfg=PRMPipelineConfig())
        # ... inside train loop:
        metrics = prm_pipeline.step(iter_idx, batch, tokenizer=tokenizer)
        # metrics is {} on skipped iters, else {"prm_loss": ..., "prm_acc": ...}

    Attributes
    ----------
    prm : ProcessRewardModel
        The model being trained online.
    trainer : PRMTrainer
        The underlying step-level trainer.
    generator : PRMSampleGenerator
        Converts RolloutBatch → samples (set lazily on first call).
    _replay : deque
        FIFO sample replay buffer.
    _n_calls : int
        Counter of how many times ``step()`` has been called (used for
        ``train_every`` gating).
    """

    def __init__(
        self,
        prm: ProcessRewardModel,
        cfg: PRMPipelineConfig | None = None,
        *,
        logger: Any = None,
    ) -> None:
        self.prm = prm
        self.cfg = cfg or PRMPipelineConfig()
        self.trainer = PRMTrainer(prm, cfg=self.cfg.prm_cfg, logger=logger)
        self.generator: PRMSampleGenerator | None = None
        self._replay: deque[PRMStepSample] = deque(
            maxlen=max(1, self.cfg.replay_capacity) if self.cfg.replay_capacity > 0 else None
        )
        self._n_calls = 0
        self._total_samples_generated = 0
        self._total_train_rounds = 0

    def step(
        self,
        iter_idx: int,
        batch: RolloutBatch,
        *,
        tokenizer: Any,
    ) -> dict[str, Any]:
        """Potentially run a PRM training round.

        Returns a metrics dict (empty when skipped).
        """
        self._n_calls += 1

        if self.cfg.train_every <= 0:
            return {}

        # Lazily build generator (tokenizer may not be available at __init__).
        if self.generator is None:
            self.generator = PRMSampleGenerator(self.cfg, tokenizer)

        # Generate fresh samples from current rollout batch.
        fresh = self.generator.generate(batch)
        self._total_samples_generated += len(fresh)

        # Add fresh samples to replay.
        if self.cfg.replay_capacity > 0:
            self._replay.extend(fresh)

        # Gate: only train every N iters.
        if iter_idx % self.cfg.train_every != 0:
            return {}

        # Compose training set from fresh + replay mix.
        train_samples = list(fresh)
        if self.cfg.replay_capacity > 0 and self._replay:
            n_replay = int(len(fresh) * self.cfg.replay_ratio)
            if n_replay > 0:
                replay_pool = list(self._replay)
                n_replay = min(n_replay, len(replay_pool))
                replayed = random.sample(replay_pool, n_replay)
                train_samples = train_samples + replayed

        if not train_samples:
            return {"prm_skipped": 1.0, "prm_no_samples": 1.0}

        random.shuffle(train_samples)
        result = self.trainer.train(train_samples)
        self._total_train_rounds += 1

        # Summarise training result.
        steps = result.get("steps", [])
        if not steps:
            return {"prm_train_rounds": float(self._total_train_rounds)}

        last = steps[-1]
        metrics: dict[str, Any] = {
            "prm_loss": float(last.get("loss", 0.0)),
            "prm_acc": float(last.get("acc", 0.0)),
            "prm_n_samples": float(len(train_samples)),
            "prm_n_fresh": float(len(fresh)),
            "prm_replay_size": float(len(self._replay)),
            "prm_train_rounds": float(self._total_train_rounds),
            "prm_total_samples": float(self._total_samples_generated),
        }
        return metrics

    def save(self, path: str) -> None:
        """Persist the PRM head weights."""
        self.trainer.save(path)

    def stats_summary(self) -> dict[str, Any]:
        """Return a summary of pipeline activity."""
        return {
            "n_calls": self._n_calls,
            "total_samples_generated": self._total_samples_generated,
            "total_train_rounds": self._total_train_rounds,
            "replay_size": len(self._replay),
        }


# ---------------------------------------------------------------------------
# Config integration helpers
# ---------------------------------------------------------------------------


def build_prm_pipeline_from_config(
    cfg_dict: dict[str, Any],
    prm_model: ProcessRewardModel,
    *,
    logger: Any = None,
) -> PRMPipeline:
    """Build a PRMPipeline from a raw config dict (e.g. from YAML).

    Expected keys (all optional, fall back to PRMPipelineConfig defaults):
      train_every, samples_per_iter, step_discount, positive_threshold,
      replay_capacity, replay_ratio, step_sep, min_steps,
      prm_lr, prm_n_epochs, prm_batch_size, prm_freeze_base, prm_grad_clip.
    """
    prm_cfg = PRMConfig(
        lr=float(cfg_dict.get("prm_lr", 1e-3)),
        n_epochs=int(cfg_dict.get("prm_n_epochs", 3)),
        batch_size=int(cfg_dict.get("prm_batch_size", 4)),
        freeze_base=bool(cfg_dict.get("prm_freeze_base", True)),
        grad_clip=float(cfg_dict.get("prm_grad_clip", 1.0)),
        step_sep=str(cfg_dict.get("step_sep", "\n\n")),
    )
    pipeline_cfg = PRMPipelineConfig(
        train_every=int(cfg_dict.get("train_every", 5)),
        samples_per_iter=int(cfg_dict.get("samples_per_iter", 64)),
        step_discount=float(cfg_dict.get("step_discount", 0.95)),
        positive_threshold=float(cfg_dict.get("positive_threshold", 0.5)),
        replay_capacity=int(cfg_dict.get("replay_capacity", 512)),
        replay_ratio=float(cfg_dict.get("replay_ratio", 0.25)),
        step_sep=str(cfg_dict.get("step_sep", "\n\n")),
        min_steps=int(cfg_dict.get("min_steps", 1)),
        prm_cfg=prm_cfg,
    )
    return PRMPipeline(prm_model, cfg=pipeline_cfg, logger=logger)
