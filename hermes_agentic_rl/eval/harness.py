"""Evaluation harness.

Given a policy backend, an env, a reward manager, and an agent-loop factory,
run N rollouts and compute aggregate metrics:
  - mean_reward, std_reward, success_rate (configurable threshold)
  - mean_turns, mean_response_len
  - per-component reward means (when available)

Output formats:
  - EvalReport dataclass
  - Markdown leaderboard (for comparing multiple models side-by-side)
  - JSON summary

Deterministic: each rollout uses ``seed_base + i`` so re-runs match.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.trainers.on_policy import AgentLoopFactory


@dataclass(slots=True)
class EvalConfig:
    n_rollouts: int = 32
    max_new_tokens: int = 16
    temperature: float = 0.0           # greedy by default for reproducibility
    seed_base: int | None = 0
    success_threshold: float = 0.5     # reward ≥ threshold counts as success


@dataclass(slots=True)
class EvalReport:
    name: str
    n_rollouts: int
    mean_reward: float
    std_reward: float
    success_rate: float
    mean_turns: float
    mean_response_len: float
    component_means: dict[str, float] = field(default_factory=dict, repr=False)
    rewards: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvalHarness:
    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        cfg: EvalConfig | None = None,
        *,
        name: str = "policy",
        agent_loop_factory: AgentLoopFactory | None = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.reward_manager = reward_manager
        self.cfg = cfg or EvalConfig()
        self.name = name
        self.agent_loop_factory = agent_loop_factory or self._default_loop

    def _default_loop(self, *, backend: LLMBackend, seed: int | None):
        return PolicyAgentLoop(
            backend=backend,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            seed=seed,
        )

    async def _run_async(self) -> EvalReport:
        await self.env.setup()
        rewards: list[float] = []
        turns_used: list[int] = []
        response_lens: list[int] = []
        component_sums: dict[str, float] = {}
        component_counts: dict[str, int] = {}
        for i in range(self.cfg.n_rollouts):
            item = await self.env.get_next_item()
            instruction = self.env.format_prompt(item)
            seed = (self.cfg.seed_base + i) if self.cfg.seed_base is not None else None
            loop = self.agent_loop_factory(backend=self.policy, seed=seed)
            traj = await RolloutManager(loop).collect(item, instruction)
            summary = await self.reward_manager.evaluate(item, traj, tool_context=None)
            rewards.append(float(summary.final_score))
            turns_used.append(int(traj.turns_used))
            rl = (traj.metadata.get("runtime") or {}).get("rl") if isinstance(
                traj.metadata.get("runtime"), dict
            ) else None
            if rl:
                response_lens.append(len(rl.get("response_ids") or []))
            for comp in summary.components:
                component_sums[comp.name] = component_sums.get(comp.name, 0.0) + comp.score
                component_counts[comp.name] = component_counts.get(comp.name, 0) + 1
        n = max(1, len(rewards))
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        std = var ** 0.5
        success = sum(1 for r in rewards if r >= self.cfg.success_threshold) / n
        comp_means = {
            k: component_sums[k] / max(1, component_counts[k]) for k in component_sums
        }
        return EvalReport(
            name=self.name,
            n_rollouts=len(rewards),
            mean_reward=mean,
            std_reward=std,
            success_rate=success,
            mean_turns=sum(turns_used) / n,
            mean_response_len=(sum(response_lens) / n) if response_lens else 0.0,
            component_means=comp_means,
            rewards=rewards,
        )

    def run(self) -> EvalReport:
        return asyncio.run(self._run_async())


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def leaderboard_markdown(reports: list[EvalReport]) -> str:
    """Render a markdown table comparing multiple eval reports."""
    if not reports:
        return "| (no reports) |\n"
    comp_keys = sorted({k for r in reports for k in r.component_means.keys()})
    header = ["name", "n", "mean_reward", "std", "success_rate", "mean_turns", "mean_len"]
    header.extend(comp_keys)
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in reports:
        row = [
            r.name,
            str(r.n_rollouts),
            f"{r.mean_reward:.4f}",
            f"{r.std_reward:.4f}",
            f"{r.success_rate:.3f}",
            f"{r.mean_turns:.2f}",
            f"{r.mean_response_len:.1f}",
        ]
        for k in comp_keys:
            row.append(f"{r.component_means.get(k, 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def save_report(report: EvalReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
