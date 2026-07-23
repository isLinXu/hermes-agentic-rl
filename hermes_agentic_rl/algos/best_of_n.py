"""Best-of-N sampling and Rejection Sampling for RL fine-tuning.

Best-of-N:
  Generate N rollouts per prompt; pick the one with the highest reward
  score (from an ORM or PRM). Returns the winning rollout + its score.
  Useful for inference-time scaling and generating preference pairs.

Rejection Sampling Fine-tuning (RSF / STaR):
  Generate N rollouts per prompt; keep only those above a reward threshold.
  Use these as SFT targets (BC on high-reward completions).
  Avoids RL instability while still steering toward high-reward behaviour.

Usage::

    bon = BestOfN(
        policy=backend,
        env=env,
        reward_manager=rm,
        cfg=BestOfNConfig(n=8, temperature=1.0),
    )
    # Inference-time: pick best completion
    winner = asyncio.run(bon.best(item))

    # Offline: generate preference pairs for DPO
    pairs = asyncio.run(bon.preference_pairs(item))

    # Rejection sampling: generate SFT dataset
    rs = RejectionSampler(bon, threshold=0.5)
    samples = asyncio.run(rs.sample(item))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.algos.base import RolloutRecord
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BestOfNResult:
    """Result from a Best-of-N sampling run."""

    item: dict[str, Any]
    winner: RolloutRecord
    winner_score: float
    all_scores: list[float]
    all_records: list[RolloutRecord]


@dataclass(slots=True)
class PreferencePair:
    """(prompt, chosen, rejected) triple for DPO training."""

    prompt_ids: list[int]
    chosen_ids: list[int]
    chosen_score: float
    rejected_ids: list[int]
    rejected_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RejectionSample:
    """High-quality rollout accepted by the rejection sampler."""

    prompt_ids: list[int]
    response_ids: list[int]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BestOfNConfig:
    n: int = 8  # number of rollouts per prompt
    temperature: float = 1.0  # sampling temperature
    max_new_tokens: int = 256
    seed_base: int = 0  # seeds: seed_base + i for i in range(n)
    min_response_tokens: int = 1  # skip empty rollouts


# ---------------------------------------------------------------------------
# BestOfN
# ---------------------------------------------------------------------------


class BestOfN:
    """Generate N rollouts per prompt and select/rank by reward.

    Works with any BaseEnv + RewardManager + LLMBackend — no changes
    to existing code required.
    """

    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        cfg: BestOfNConfig | None = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.reward_manager = reward_manager
        self.cfg = cfg or BestOfNConfig()

    async def _single_rollout(
        self, item: dict[str, Any], seed: int
    ) -> tuple[RolloutRecord, float] | None:
        """Run one rollout and score it. Returns (record, score) or None."""
        from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop
        from hermes_agentic_rl.mdp.state_encoder import PromptStateEncoder

        prompt_text = self.env.format_prompt(item)
        encoder = PromptStateEncoder(self.policy.tokenizer)
        obs = encoder.encode(item)
        prompt_ids: list[int] = list(obs.prompt_ids)

        loop = PolicyAgentLoop(
            backend=self.policy,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            seed=seed,
        )
        # PolicyAgentLoop.run() returns a raw dict; wrap it into a Trajectory.
        raw: dict[str, Any] = await loop.run(prompt_text)

        traj = Trajectory(
            task_id=str(item.get("task_id", seed)),
            prompt=prompt_text,
            steps=[],
            final_output=raw.get("final_output"),
            finished_naturally=bool(raw.get("finished_naturally", False)),
            turns_used=int(raw.get("turns_used", 1)),
            metadata=raw.get("metadata", {}),
        )

        rt = traj.metadata.get("runtime") or {}
        rl_meta = rt.get("rl") if isinstance(rt, dict) else None
        if rl_meta is None:
            return None
        response_ids = list(rl_meta.get("response_ids") or [])
        if len(response_ids) < self.cfg.min_response_tokens:
            return None

        summary = await self.reward_manager.evaluate(item, traj, None)
        score = float(getattr(summary, "final_score", 0.0))

        record = RolloutRecord(
            prompt_ids=list(rl_meta.get("prompt_ids") or prompt_ids),
            response_ids=response_ids,
            old_logprobs=list(rl_meta.get("old_logprobs") or []),
            reward=score,
            group_id=str(seed),
            metadata={
                "seed": seed,
                "old_seq_logprob": float(sum(rl_meta.get("old_logprobs") or [])),
                **traj.metadata,
            },
        )
        return record, score

    async def run_all(self, item: dict[str, Any]) -> BestOfNResult:
        """Generate N rollouts, rank by reward, return structured result."""
        cfg = self.cfg
        tasks = [self._single_rollout(item, cfg.seed_base + i) for i in range(cfg.n)]
        results = await asyncio.gather(*tasks)
        valid = [(rec, score) for r in results if r is not None for rec, score in [r]]

        if not valid:
            # Fallback: return empty winner
            empty_record = RolloutRecord(
                prompt_ids=[],
                response_ids=[],
                old_logprobs=[],
                reward=0.0,
                group_id="bon_empty",
            )
            return BestOfNResult(
                item=item,
                winner=empty_record,
                winner_score=0.0,
                all_scores=[],
                all_records=[],
            )

        valid.sort(key=lambda x: x[1], reverse=True)
        winner_rec, winner_score = valid[0]
        return BestOfNResult(
            item=item,
            winner=winner_rec,
            winner_score=winner_score,
            all_scores=[s for _, s in valid],
            all_records=[r for r, _ in valid],
        )

    async def best(self, item: dict[str, Any]) -> BestOfNResult:
        """Convenience wrapper: run_all and return the result."""
        return await self.run_all(item)

    async def preference_pairs(self, item: dict[str, Any]) -> list[PreferencePair]:
        """Generate preference pairs (chosen, rejected) for DPO.

        Pairs the highest-scoring rollout against each lower-scoring one.
        Returns up to N-1 pairs.
        """
        result = await self.run_all(item)
        if len(result.all_records) < 2:
            return []

        chosen = result.all_records[0]
        chosen_score = result.all_scores[0]
        pairs: list[PreferencePair] = []
        for i in range(1, len(result.all_records)):
            rejected = result.all_records[i]
            rejected_score = result.all_scores[i]
            if chosen_score <= rejected_score:
                continue  # no clear preference
            pairs.append(
                PreferencePair(
                    prompt_ids=chosen.prompt_ids,
                    chosen_ids=chosen.response_ids,
                    chosen_score=chosen_score,
                    rejected_ids=rejected.response_ids,
                    rejected_score=rejected_score,
                    metadata={"item": item},
                )
            )
        return pairs


# ---------------------------------------------------------------------------
# RejectionSampler
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RejectionSamplerConfig:
    threshold: float = 0.5  # minimum reward to accept
    max_accept: int = 1  # max accepted samples per prompt
    percentile_mode: bool = False
    # When percentile_mode=True, threshold is interpreted as the fraction
    # of rollouts to accept (e.g. 0.25 = top 25%).


class RejectionSampler:
    """Accept rollouts above a reward threshold for SFT (RSF / STaR).

    Works as a thin wrapper around BestOfN.
    """

    def __init__(
        self,
        bon: BestOfN,
        cfg: RejectionSamplerConfig | None = None,
    ) -> None:
        self.bon = bon
        self.cfg = cfg or RejectionSamplerConfig()

    async def sample(self, item: dict[str, Any]) -> list[RejectionSample]:
        """Return high-reward rollouts for this item."""
        result = await self.bon.run_all(item)
        records = result.all_records
        scores = result.all_scores

        if not records:
            return []

        # Determine threshold
        threshold = self.cfg.threshold
        if self.cfg.percentile_mode and scores:
            sorted_scores = sorted(scores, reverse=True)
            cutoff_idx = max(0, int(len(sorted_scores) * self.cfg.threshold) - 1)
            threshold = sorted_scores[cutoff_idx]

        accepted: list[RejectionSample] = []
        for rec, score in zip(records, scores, strict=False):
            if score >= threshold and rec.response_ids:
                accepted.append(
                    RejectionSample(
                        prompt_ids=rec.prompt_ids,
                        response_ids=rec.response_ids,
                        score=score,
                        metadata={"item": item, "bon_score": score},
                    )
                )
            if len(accepted) >= self.cfg.max_accept:
                break
        return accepted

    async def build_sft_dataset(self, items: list[dict[str, Any]]) -> list[RejectionSample]:
        """Run rejection sampling over a list of items.

        Returns all accepted samples suitable for BC / SFT training.
        """
        all_samples: list[RejectionSample] = []
        for item in items:
            samples = await self.sample(item)
            all_samples.extend(samples)
        return all_samples
