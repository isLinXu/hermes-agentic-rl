from __future__ import annotations

from collections import defaultdict
from typing import Any

from hermes_agentic_rl.offline.replay_buffer import DPOPair, TrainSample


def build_preference_pairs_from_samples(
    samples: list[TrainSample],
    *,
    min_reward_gap: float = 0.25,
    max_pairs_per_prompt: int = 1,
) -> list[DPOPair]:
    grouped: dict[tuple[int, ...], list[TrainSample]] = defaultdict(list)
    for sample in samples:
        grouped[tuple(sample.prompt_ids)].append(sample)

    pairs: list[DPOPair] = []
    for prompt_key, group in grouped.items():
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda s: float(s.reward), reverse=True)
        added = 0
        for chosen in ranked:
            for rejected in reversed(ranked):
                if chosen is rejected:
                    continue
                reward_gap = float(chosen.reward) - float(rejected.reward)
                if reward_gap < min_reward_gap:
                    continue
                if list(chosen.response_ids) == list(rejected.response_ids):
                    continue
                pairs.append(
                    DPOPair(
                        prompt_ids=list(prompt_key),
                        chosen_ids=list(chosen.response_ids),
                        rejected_ids=list(rejected.response_ids),
                        metadata={
                            "chosen_reward": float(chosen.reward),
                            "rejected_reward": float(rejected.reward),
                            "reward_gap": reward_gap,
                            "chosen_metadata": dict(chosen.metadata),
                            "rejected_metadata": dict(rejected.metadata),
                        },
                    )
                )
                added += 1
                if added >= max(1, max_pairs_per_prompt):
                    break
            if added >= max(1, max_pairs_per_prompt):
                break
    return pairs


def records_to_train_samples(
    records: list[dict[str, Any]],
    *,
    min_reward: float = -1e9,
) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for record in records:
        if "prompt_ids" not in record or "response_ids" not in record:
            continue
        reward = float(record.get("reward", 0.0))
        if reward < min_reward:
            continue
        samples.append(
            TrainSample(
                prompt_ids=list(record["prompt_ids"]),
                response_ids=list(record["response_ids"]),
                reward=reward,
                advantage=record.get("advantage"),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return samples
