from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward


@dataclass(slots=True)
class ContextBenchmarkConfig:
    dataset_path: str | None = None
    dataset_size: int = 8
    seed: int = 0
    noise_blocks: int = 6
    max_response_chars: int = 360
    required_weight: float = 0.35
    constraint_weight: float = 0.25
    tool_summary_weight: float = 0.20
    distractor_weight: float = 0.10
    compression_weight: float = 0.10


class ContextBenchmarkReward(BaseReward):
    name = "context_benchmark_reward"

    def __init__(
        self, config: ContextBenchmarkConfig | None = None, *, weight: float = 1.0
    ) -> None:
        self.config = config or ContextBenchmarkConfig()
        self.weight = float(weight)

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context
        output = str(trajectory.final_output or "")
        lowered = _normalize_text(output)

        required = _string_list(item.get("required_facts"))
        constraints = _string_list(item.get("constraints"))
        tool_facts = _string_list(item.get("tool_summary_facts"))
        forbidden = _string_list(item.get("forbidden_facts"))
        max_chars = int(item.get("max_response_chars") or self.config.max_response_chars)

        required_recall = _hit_rate(lowered, required)
        constraint_satisfaction = _hit_rate(lowered, constraints)
        tool_retention = _hit_rate(lowered, tool_facts)
        forbidden_hits = _hit_count(lowered, forbidden)
        distractor_avoidance = (
            1.0 if not forbidden else max(0.0, 1.0 - forbidden_hits / len(forbidden))
        )
        compression_ok = 1.0 if output and len(output) <= max_chars else 0.0
        precision_denominator = (
            _hit_count(lowered, required)
            + _hit_count(lowered, constraints)
            + _hit_count(lowered, tool_facts)
            + forbidden_hits
        )
        precision = (
            (
                _hit_count(lowered, required)
                + _hit_count(lowered, constraints)
                + _hit_count(lowered, tool_facts)
            )
            / precision_denominator
            if precision_denominator
            else 0.0
        )

        cfg = self.config
        score = (
            required_recall * cfg.required_weight
            + constraint_satisfaction * cfg.constraint_weight
            + tool_retention * cfg.tool_summary_weight
            + distractor_avoidance * cfg.distractor_weight
            + compression_ok * cfg.compression_weight
        )
        metadata = {
            "context_required_fact_recall": required_recall,
            "context_constraint_satisfaction": constraint_satisfaction,
            "context_tool_summary_retention": tool_retention,
            "context_distractor_avoidance": distractor_avoidance,
            "context_precision": precision,
            "context_compression_ok": compression_ok,
            "context_response_chars": len(output),
            "context_prompt_chars": len(str(item.get("prompt", ""))),
            "context_noise_blocks": len(_string_list(item.get("noise_blocks"))),
            "context_required_facts": len(required),
            "context_forbidden_hits": forbidden_hits,
        }
        return RewardResult(
            name=self.name,
            score=round(float(score), 6),
            reason=(
                f"required={required_recall:.2f} constraints={constraint_satisfaction:.2f} "
                f"tool={tool_retention:.2f} distractor={distractor_avoidance:.2f} "
                f"compression={compression_ok:.2f}"
            ),
            weight=self.weight,
            metadata=metadata,
        )


class ContextBenchmarkEnv(BaseEnv):
    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        config: ContextBenchmarkConfig | None = None,
    ) -> None:
        if not items:
            raise ValueError("ContextBenchmarkEnv requires at least one item")
        self.items = [dict(item) for item in items]
        self.config = config or ContextBenchmarkConfig()
        self._reward = ContextBenchmarkReward(self.config)
        self._idx = 0

    @property
    def reward_component(self) -> ContextBenchmarkReward:
        return self._reward

    async def setup(self) -> None:
        self._idx = 0

    async def get_next_item(self) -> dict[str, Any]:
        item = self.items[self._idx % len(self.items)]
        self._idx += 1
        return dict(item)

    def format_prompt(self, item: dict[str, Any]) -> str:
        if isinstance(item.get("prompt"), str) and str(item["prompt"]).strip():
            return str(item["prompt"])

        task = str(item.get("task") or "Answer from the context only.")
        required = _string_list(item.get("required_facts"))
        constraints = _string_list(item.get("constraints"))
        tool_facts = _string_list(item.get("tool_summary_facts"))
        forbidden = _string_list(item.get("forbidden_facts"))
        noise = _string_list(item.get("noise_blocks"))
        max_chars = int(item.get("max_response_chars") or self.config.max_response_chars)

        sections = [
            "You are evaluating prompt-context control for a Hermes-style agent.",
            "Use only the relevant context. Ignore distractors and stale facts.",
            "",
            f"Task: {task}",
            "",
            "Relevant memory facts:",
            *[f"- {fact}" for fact in required],
            "",
            "User constraints:",
            *[f"- {constraint}" for constraint in constraints],
            "",
            "Tool result summary that must be preserved:",
            *[f"- {fact}" for fact in tool_facts],
            "",
            "Distractor or stale context that must not appear in the final answer:",
            *[f"- {fact}" for fact in forbidden],
            "",
            "Additional noisy context:",
            *[f"- {block}" for block in noise],
            "",
            f"Final answer must be concise, at most {max_chars} characters, "
            "and include the relevant facts.",
        ]
        return "\n".join(sections)

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return [await self._reward.evaluate(item, trajectory, tool_context)]

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        target = item.get("target_response")
        if not isinstance(target, str) or not target.strip():
            return []
        return [
            SupervisedSample(
                instruction=self.format_prompt(item),
                response=target.strip(),
                metadata={"task_id": item.get("task_id"), "capability_axis": "prompt_context"},
            )
        ]

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> ContextBenchmarkEnv:
        cfg = ContextBenchmarkConfig(
            dataset_path=str(raw["dataset_path"]) if raw.get("dataset_path") else None,
            dataset_size=int(raw.get("dataset_size", 8)),
            seed=int(raw.get("dataset_seed", raw.get("seed", 0))),
            noise_blocks=int(raw.get("noise_blocks", 6)),
            max_response_chars=int(raw.get("max_response_chars", 360)),
            required_weight=float(raw.get("required_weight", 0.35)),
            constraint_weight=float(raw.get("constraint_weight", 0.25)),
            tool_summary_weight=float(raw.get("tool_summary_weight", 0.20)),
            distractor_weight=float(raw.get("distractor_weight", 0.10)),
            compression_weight=float(raw.get("compression_weight", 0.10)),
        )
        items = (
            _load_jsonl(cfg.dataset_path)
            if cfg.dataset_path
            else build_context_benchmark_dataset(
                n=cfg.dataset_size,
                seed=cfg.seed,
                noise_blocks=cfg.noise_blocks,
                max_response_chars=cfg.max_response_chars,
            )
        )
        return cls(items, config=cfg)


def build_context_benchmark_dataset(
    *,
    n: int = 8,
    seed: int = 0,
    noise_blocks: int = 6,
    max_response_chars: int = 360,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    projects = [
        ("atlas", "blue", "CSV parser", "latency regression", "ship summary first"),
        ("nova", "green", "OAuth callback", "token refresh bug", "include rollback note"),
        ("ember", "orange", "vector index", "duplicate chunk issue", "mention validation split"),
        ("lumen", "silver", "terminal runner", "stderr capture gap", "keep command names exact"),
    ]
    stale_facts = [
        "use the deprecated purple deployment",
        "send results to the old staging bucket",
        "mark the task as blocked",
        "ignore the tool warning",
        "switch the owner to legacy ops",
    ]
    noise_templates = [
        "Historical note {idx}: a different project used {color} labels "
        "and unrelated release notes.",
        "Chat fragment {idx}: someone mentioned lunch, roadmap guesses, "
        "and non-actionable speculation.",
        "Old tool output {idx}: status=unknown, owner=archive, priority=low.",
        "Reference {idx}: this paragraph is intentionally irrelevant filler for context pressure.",
    ]

    items: list[dict[str, Any]] = []
    for idx in range(max(1, n)):
        project, color, component, issue, constraint = projects[idx % len(projects)]
        required = [
            f"project {project}",
            f"label {color}",
            component,
        ]
        tool_facts = [
            f"tool found {issue}",
            f"owner team-{idx % 3}",
        ]
        constraints = [
            constraint,
            "do not mention stale context",
        ]
        forbidden = rng.sample(stale_facts, k=min(2, len(stale_facts)))
        noise = [
            rng.choice(noise_templates).format(
                idx=noise_idx, color=rng.choice(["red", "teal", "black"])
            )
            for noise_idx in range(noise_blocks)
        ]
        task = (
            f"Summarize the latest state for {project} using the relevant memory, "
            "the current tool summary, and the user constraints."
        )
        target = (
            f"Project {project} uses label {color}. The relevant component is {component}; "
            f"the tool found {issue} and owner team-{idx % 3}. {constraint}."
        )
        item = {
            "task_id": f"context-benchmark-{idx}",
            "task": task,
            "required_facts": required,
            "constraints": constraints,
            "tool_summary_facts": tool_facts,
            "forbidden_facts": forbidden,
            "noise_blocks": noise,
            "target_response": target,
            "max_response_chars": max_response_chars,
            "category": "context_benchmark",
        }
        item["prompt"] = ContextBenchmarkEnv([item], config=ContextBenchmarkConfig()).format_prompt(
            item
        )
        items.append(item)
    return items


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _hit_count(text: str, facts: list[str]) -> int:
    return sum(1 for fact in facts if _normalize_text(fact) in text)


def _hit_rate(text: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    return _hit_count(text, facts) / len(facts)
