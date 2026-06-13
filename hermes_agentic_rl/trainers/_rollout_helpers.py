"""Pure-function helpers extracted from on_policy.py.

These have zero dependency on OnPolicyTrainer state and are imported back
into on_policy.py so existing call sites are unchanged.
"""

from __future__ import annotations

import statistics
from typing import Any

import torch

from hermes_agentic_rl.algos.base import RolloutBatch
from hermes_agentic_rl.core.types import RolloutStep, Trajectory

# ---------------------------------------------------------------------------
# Trajectory / metadata extraction helpers
# ---------------------------------------------------------------------------


def extract_rl(trajectory: Trajectory) -> dict[str, Any] | None:
    """Extract the RL metadata dict from a trajectory, checking both layouts."""
    runtime_block = trajectory.metadata.get("runtime")
    if isinstance(runtime_block, dict):
        rl = runtime_block.get("rl")
        if isinstance(rl, dict):
            return rl
    rl = trajectory.metadata.get("rl")
    if isinstance(rl, dict):
        return rl
    return None


def extract_next_state(trajectory: Trajectory) -> str | None:
    """Pull the next-state signal from a trajectory's runtime metadata.

    Mirrors ``NextStatePRMComponent`` which reads
    ``trajectory.metadata["runtime"]["next_state"]``. Falls back to a top-level
    ``next_state`` key. Returns None when no textual signal is present.
    """
    runtime_block = trajectory.metadata.get("runtime")
    if isinstance(runtime_block, dict):
        val = runtime_block.get("next_state")
        if isinstance(val, str) and val.strip():
            return val
    val = trajectory.metadata.get("next_state")
    if isinstance(val, str) and val.strip():
        return val
    return None


def rollout_temperature_from_meta(
    rl_meta: dict[str, Any],
    *,
    fallback: float,
) -> float:
    raw = rl_meta.get("temperature", fallback)
    if isinstance(raw, bool):
        return float(fallback)
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(fallback)


# ---------------------------------------------------------------------------
# Trajectory builder for batched single-turn rollouts
# ---------------------------------------------------------------------------


def batch_single_turn_trajectory(
    *,
    item: dict[str, Any],
    instruction: str,
    response_text: str,
    prompt_ids: list[int],
    response_ids: list[int],
    old_logprobs: list[float],
    temperature: float,
    finished: bool,
) -> Trajectory:
    """Build a single-turn trajectory for batched rollout collection.

    Batched generation bypasses RolloutManager, so we must populate at least
    one :class:`RolloutStep` so trajectory-level rewards (tool-call fidelity,
    turn discount, conditional gating on ``traj.steps``) behave like the
    agent-loop path.
    """
    return Trajectory(
        task_id=item["task_id"],
        prompt=instruction,
        steps=[
            RolloutStep(
                turn_index=0,
                assistant_message=response_text,
            )
        ],
        final_output=response_text,
        finished_naturally=finished,
        turns_used=1,
        metadata={
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response_text},
            ],
            "runtime": {
                "runtime": "policy_agent_loop",
                "prompt": instruction,
                "rl": {
                    "prompt_ids": list(prompt_ids),
                    "response_ids": list(response_ids),
                    "old_logprobs": list(old_logprobs),
                    "temperature": temperature,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# Record/component payload helpers
# ---------------------------------------------------------------------------


def sanitize_metric_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return cleaned or "metric"


def reward_component_payload(component: Any) -> dict[str, Any]:
    payload = {
        "name": getattr(component, "name", "reward"),
        "score": getattr(component, "score", 0.0),
        "weight": getattr(component, "weight", 1.0),
    }
    metadata = getattr(component, "metadata", None)
    if isinstance(metadata, dict):
        numeric_metadata = {
            sanitize_metric_name(str(key)): float(value)
            for key, value in metadata.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric_metadata:
            payload["metadata"] = numeric_metadata
    return payload


def turn_group_id(prompt_group_id: str, turn_index: int) -> str:
    return f"{prompt_group_id}::turn:{turn_index}"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def series_stats(
    values: list[float],
    *,
    include_mean: bool = True,
) -> dict[str, float]:
    if not values:
        return {}
    summary: dict[str, float] = {
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }
    if include_mean:
        summary["mean"] = sum(values) / len(values)
    return summary


def grad_l2_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        total += float((grad.detach().float() ** 2).sum().item())
    return total ** 0.5


def param_l2_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for param in params:
        total += float((param.detach().float() ** 2).sum().item())
    return total ** 0.5


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Shallow dataclass-to-dict for checkpoint config snapshot."""
    from pathlib import Path

    out: dict[str, Any] = {}
    for name in getattr(cfg, "__slots__", []) or []:
        try:
            v = getattr(cfg, name)
        except AttributeError:
            continue
        if isinstance(v, Path):
            out[name] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[name] = v
        else:
            out[name] = repr(v)
    return out


# ---------------------------------------------------------------------------
# Batch metadata summarizer
# ---------------------------------------------------------------------------


def summarize_batch_metadata(batch: RolloutBatch) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    rewards = [float(rec.reward) for rec in batch.records]
    reward_stats = series_stats(rewards, include_mean=False)
    summary.update({f"reward_{key}": value for key, value in reward_stats.items()})

    groups = batch.by_group()
    if groups:
        summary["n_groups"] = len(groups)
        group_size_stats = series_stats([float(len(rows)) for rows in groups.values()])
        if group_size_stats:
            summary["records_per_group"] = group_size_stats
        group_reward_std = [
            series_stats([float(rec.reward) for rec in rows], include_mean=False).get("std", 0.0)
            for rows in groups.values()
        ]
        group_reward_std_stats = series_stats(group_reward_std)
        if group_reward_std_stats:
            summary["group_reward_std"] = group_reward_std_stats

    # Per-stream breakdown (multi-stream unified training).
    stream_rewards: dict[int, list[float]] = {}
    for rec in batch.records:
        lvl = rec.metadata.get("stream_level")
        if isinstance(lvl, int) and not isinstance(lvl, bool):
            stream_rewards.setdefault(lvl, []).append(float(rec.reward))
    if stream_rewards:
        summary["n_streams"] = len(stream_rewards)
        total_stream_recs = sum(len(v) for v in stream_rewards.values())
        for lvl, rws in sorted(stream_rewards.items()):
            summary[f"stream/{lvl}/count"] = float(len(rws))
            summary[f"stream/{lvl}/share"] = float(len(rws)) / total_stream_recs
            summary[f"stream/{lvl}/mean_reward"] = sum(rws) / len(rws)
            std = series_stats(rws, include_mean=False).get("std")
            if std is not None:
                summary[f"stream/{lvl}/reward_std"] = std

    prompt_tokens = [float(len(rec.prompt_ids)) for rec in batch.records]
    prompt_stats = series_stats(prompt_tokens)
    if prompt_stats:
        summary["prompt_tokens"] = prompt_stats

    response_tokens = [float(len(rec.response_ids)) for rec in batch.records]
    response_stats = series_stats(response_tokens)
    if response_stats:
        summary["response_tokens"] = response_stats

    generation_metrics = {
        "final_output_chars": [
            float(rec.metadata["final_output_chars"])
            for rec in batch.records
            if isinstance(rec.metadata.get("final_output_chars"), (int, float))
        ],
        "turns_used": [
            float(rec.metadata["turns_used"])
            for rec in batch.records
            if isinstance(rec.metadata.get("turns_used"), (int, float))
        ],
        "tool_calls_count": [
            float(rec.metadata["tool_calls_count"])
            for rec in batch.records
            if isinstance(rec.metadata.get("tool_calls_count"), (int, float))
        ],
        "tool_results_count": [
            float(rec.metadata["tool_results_count"])
            for rec in batch.records
            if isinstance(rec.metadata.get("tool_results_count"), (int, float))
        ],
        "rollout_temperature": [
            float(rec.metadata["rollout_temperature"])
            for rec in batch.records
            if isinstance(rec.metadata.get("rollout_temperature"), (int, float))
            and not isinstance(rec.metadata.get("rollout_temperature"), bool)
        ],
    }
    for key, values in generation_metrics.items():
        stats = series_stats(values)
        if stats:
            summary[key] = stats

    finished_naturally = [
        1.0 if bool(rec.metadata["finished_naturally"]) else 0.0
        for rec in batch.records
        if "finished_naturally" in rec.metadata
    ]
    if finished_naturally:
        summary["finished_naturally_rate"] = sum(finished_naturally) / len(finished_naturally)

    reward_component_scores: dict[str, list[float]] = {}
    reward_component_weights: dict[str, list[float]] = {}
    reward_component_metadata: dict[str, dict[str, list[float]]] = {}
    reward_summary_numeric: dict[str, list[float]] = {}
    for rec in batch.records:
        components = rec.metadata.get("reward_components")
        if isinstance(components, list):
            for component in components:
                if not isinstance(component, dict):
                    continue
                name = sanitize_metric_name(str(component.get("name", "reward")))
                score = component.get("score")
                if isinstance(score, (int, float)):
                    reward_component_scores.setdefault(name, []).append(float(score))
                weight = component.get("weight")
                if isinstance(weight, (int, float)):
                    reward_component_weights.setdefault(name, []).append(float(weight))
                metadata = component.get("metadata")
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            safe_key = sanitize_metric_name(str(key))
                            reward_component_metadata.setdefault(name, {}).setdefault(
                                safe_key, []
                            ).append(float(value))
        reward_meta = rec.metadata.get("reward_summary_metadata")
        if isinstance(reward_meta, dict):
            for key, value in reward_meta.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    reward_summary_numeric.setdefault(
                        sanitize_metric_name(str(key)), []
                    ).append(float(value))

    if reward_component_scores:
        summary["reward_components"] = {}
        for name, values in sorted(reward_component_scores.items()):
            component_summary = series_stats(values)
            weight_values = reward_component_weights.get(name, [])
            if weight_values:
                component_summary["weight_mean"] = sum(weight_values) / len(weight_values)
            summary["reward_components"][name] = component_summary

    if reward_component_metadata:
        summary["reward_component_metadata"] = {
            component_name: {
                key: series_stats(values)
                for key, values in sorted(metadata.items())
                if values
            }
            for component_name, metadata in sorted(reward_component_metadata.items())
        }

    if reward_summary_numeric:
        summary["reward_summary"] = {
            key: series_stats(values)
            for key, values in sorted(reward_summary_numeric.items())
            if values
        }

    turn_credit_rows: list[dict[str, Any]] = []
    for rec in batch.records:
        turn_credit = rec.metadata.get("turn_credit")
        if isinstance(turn_credit, dict):
            turn_credit_rows.append(turn_credit)
    if turn_credit_rows:
        numeric_keys = [
            "reward",
            "final_component",
            "local_component",
            "judge_component",
            "teacher_component",
            "weighted_final_component",
            "weighted_local_component",
        ]
        summary["n_turn_records"] = len(turn_credit_rows)
        summary["turn_credit"] = {}
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in turn_credit_rows
                if isinstance(row.get(key), (int, float))
            ]
            stats = series_stats(values)
            if stats:
                summary["turn_credit"][key] = stats
                for stat_name, stat_val in stats.items():
                    summary[f"turn_credit_{key}_{stat_name}"] = stat_val

        turn_indices = [
            float(rec.metadata["turn_index"])
            for rec in batch.records
            if isinstance(rec.metadata.get("turn_index"), int)
        ]
        turn_index_stats = series_stats(turn_indices)
        if turn_index_stats:
            summary["turn_index"] = turn_index_stats

        rollout_final_rewards = [
            float(rec.metadata["rollout_final_reward"])
            for rec in batch.records
            if isinstance(rec.metadata.get("rollout_final_reward"), (int, float))
        ]
        rollout_reward_stats = series_stats(rollout_final_rewards)
        if rollout_reward_stats:
            summary["rollout_final_reward"] = rollout_reward_stats

    return summary
