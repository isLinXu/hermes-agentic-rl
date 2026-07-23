"""Multi-agent environment support.

Extends BaseEnv to support N concurrent agents that share state,
communicate via a message board, and receive individual or team rewards.

Architecture:
  - MultiAgentEnv: base class with agent registry and message board.
  - CoopLetterCounting: cooperative variant of LetterCountingEnv where
    Agent A counts one set of letters and Agent B counts another set,
    then they agree on a combined answer.
  - DebateEnv: two agents debate (proposer / skeptic), judge evaluates.

Agent roles:
  - Each agent is identified by an agent_id (string).
  - Observations: the shared env state + the message board visible to the agent.
  - Actions: text response + optional structured message to other agents.
  - Rewards: can be individual (based on own accuracy) or shared (team reward).

Integration with OnPolicyTrainer:
  MultiAgentEnv wraps each agent's experience as a separate RolloutRecord
  so existing single-agent trainers work without modification. The trainer
  receives N records (one per agent) per env step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample


def _empty_trajectory() -> Trajectory:
    return Trajectory(
        task_id="",
        prompt="",
        steps=[],
        final_output="",
        finished_naturally=False,
        turns_used=0,
    )

# ---------------------------------------------------------------------------
# Message board (simple shared state for agent-to-agent communication)
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    sender: str
    recipient: str | None  # None = broadcast
    content: str
    step: int


class MessageBoard:
    """Shared communication channel for multi-agent envs."""

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []
        self._step: int = 0

    def post(self, sender: str, content: str, recipient: str | None = None) -> None:
        self._messages.append(AgentMessage(sender, recipient, content, self._step))

    def read(self, agent_id: str) -> list[AgentMessage]:
        """Return messages visible to agent_id (sent to them or broadcast)."""
        return [m for m in self._messages if m.recipient is None or m.recipient == agent_id]

    def advance(self) -> None:
        self._step += 1

    def clear(self) -> None:
        self._messages.clear()
        self._step = 0


# ---------------------------------------------------------------------------
# Multi-agent item: N agents each get an observation
# ---------------------------------------------------------------------------


@dataclass
class MultiAgentItem:
    """One environment step for N agents.

    base_item: the raw env item (same as single-agent).
    agent_observations: per-agent customized observations (prompt text).
    shared_state: any state visible to all agents.
    """

    base_item: dict[str, Any]
    agent_observations: dict[str, str]  # agent_id → observation text
    shared_state: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MultiAgentEnv base
# ---------------------------------------------------------------------------


class MultiAgentEnv(BaseEnv):
    """Base class for multi-agent environments.

    Subclasses must implement:
      - get_agents() → list[str]: return agent IDs.
      - get_next_multi_item() → MultiAgentItem: the shared env item with
          per-agent observations.
      - compute_multi_reward(item, trajectories, board) → dict[str, RewardResult]:
          per-agent reward results.

    The default get_next_item() / compute_reward() delegate to the multi-agent
    API by treating agent_id = "agent_0".
    """

    def get_agents(self) -> list[str]:
        raise NotImplementedError

    async def get_next_multi_item(self) -> MultiAgentItem:
        raise NotImplementedError

    async def compute_multi_reward(
        self,
        item: MultiAgentItem,
        trajectories: dict[str, Trajectory],
        board: MessageBoard,
        tool_context: Any,
    ) -> dict[str, RewardResult]:
        raise NotImplementedError

    # ── Single-agent compatibility shim ──────────────────────────────────

    async def get_next_item(self) -> dict[str, Any]:
        multi = await self.get_next_multi_item()
        agents = self.get_agents()
        agent_id = agents[0] if agents else "agent_0"
        obs = multi.agent_observations.get(agent_id, "")
        return {"__multi__": multi, "__agent_id__": agent_id, "instruction": obs, **multi.base_item}

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        multi: MultiAgentItem = item.get("__multi__") or MultiAgentItem(
            base_item=item, agent_observations={}
        )
        agent_id = item.get("__agent_id__", "agent_0")
        board = MessageBoard()
        results = await self.compute_multi_reward(
            multi, {agent_id: trajectory}, board, tool_context
        )
        r = results.get(agent_id)
        return [r] if r else []

    def format_prompt(self, item: dict[str, Any]) -> str:
        return str(item.get("instruction", ""))

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        return []

    def observe(self, score: float) -> None:
        pass

    async def setup(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# CoopLetterCounting: cooperative multi-agent letter counting
# ---------------------------------------------------------------------------


class CoopLetterCountingEnv(MultiAgentEnv):
    """Two-agent cooperative letter counting.

    Agent A counts the first half of the target letters.
    Agent B counts the second half.
    Team reward = 1.0 only when both agents are correct.
    Individual partial reward = partial accuracy of each agent's answers.

    This tests cooperative credit assignment: both agents must cooperate
    for the team reward, but each also has individual partial signal.
    """

    def __init__(self, seed: int = 42, team_reward_weight: float = 0.5) -> None:
        import random

        from hermes_agentic_rl.envs.letter_counting import (
            LetterCountingConfig,
            LetterCountingEnv,
        )

        self._inner = LetterCountingEnv(LetterCountingConfig(seed=seed))
        self._rng = random.Random(seed)
        self.team_reward_weight = team_reward_weight

    def get_agents(self) -> list[str]:
        return ["agent_A", "agent_B"]

    async def setup(self) -> None:
        await self._inner.setup()

    async def close(self) -> None:
        await self._inner.close()

    async def get_next_multi_item(self) -> MultiAgentItem:
        item = await self._inner.get_next_item()
        targets = item.get("target_letters", [])
        correct = item.get("correct_counts", {})

        # Split targets between the two agents
        mid = max(1, len(targets) // 2)
        targets_A = targets[:mid]
        targets_B = targets[mid:] or targets[:1]  # B always has at least one

        text = item.get("text", "")
        counts_A = {ch: correct.get(ch, 0) for ch in targets_A}
        counts_B = {ch: correct.get(ch, 0) for ch in targets_B}

        def fmt_obs(agent_letters: list[str], counts: dict[str, int]) -> str:
            import json as _json

            letters_str = ", ".join(f"'{ch}'" for ch in agent_letters)
            if len(agent_letters) == 1:
                return (
                    f"You are part of a two-agent team. Your task: "
                    f"count how many '{agent_letters[0]}'s are in the string.\n\n"
                    f"String: {text}\n\n"
                    f"Provide your answer: <answer>{agent_letters[0]}_count</answer>"
                )
            example = _json.dumps({ch: 0 for ch in agent_letters})
            return (
                f"You are part of a two-agent team. Your task: "
                f"count letters {letters_str} in the string.\n\n"
                f"String: {text}\n\n"
                f"Provide your answer as JSON: <answer>{example}</answer>"
            )

        return MultiAgentItem(
            base_item=item,
            agent_observations={
                "agent_A": fmt_obs(targets_A, counts_A),
                "agent_B": fmt_obs(targets_B, counts_B),
            },
            shared_state={
                "targets_A": targets_A,
                "targets_B": targets_B,
                "correct_A": counts_A,
                "correct_B": counts_B,
            },
        )

    async def compute_multi_reward(
        self,
        item: MultiAgentItem,
        trajectories: dict[str, Trajectory],
        board: MessageBoard,
        tool_context: Any,
    ) -> dict[str, RewardResult]:
        import json
        import re

        shared = item.shared_state
        results: dict[str, RewardResult] = {}

        def _partial(text: str, targets: list[str], correct: dict) -> tuple[float, str]:
            m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if m is None:
                return 0.0, "no_tag"
            content = m.group(1).strip()
            if len(targets) == 1:
                try:
                    pred = int(content)
                    exp = correct[targets[0]]
                    return max(0.0, 1.0 - abs(pred - exp) / (exp + 1)), f"pred={pred}"
                except (ValueError, TypeError):
                    return 0.0, "parse_fail"
            else:
                try:
                    pd = json.loads(content)
                    if not isinstance(pd, dict):
                        raise ValueError
                except (json.JSONDecodeError, ValueError):
                    return 0.2, "bad_json"  # small credit for reaching JSON
                credits = []
                for ch in targets:
                    exp = correct.get(ch, 0)
                    try:
                        p = int(pd.get(ch, -999))
                        credits.append(max(0.0, 1.0 - abs(p - exp) / (exp + 1)))
                    except (ValueError, TypeError):
                        credits.append(0.0)
                return sum(credits) / len(credits), f"mean={sum(credits) / len(credits):.2f}"

        for agent_id, traj in trajectories.items():
            is_A = agent_id == "agent_A"
            targets = shared.get("targets_A") if is_A else shared.get("targets_B")
            correct = shared.get("correct_A") if is_A else shared.get("correct_B")
            resp = traj.final_output or ""
            individual_score, reason = _partial(resp, targets or [], correct or {})

            # Team bonus: if both agents answered perfectly
            team_score = 0.0
            other_traj = trajectories.get("agent_B" if is_A else "agent_A")
            if other_traj and individual_score >= 0.99:
                other_targets = shared.get("targets_B") if is_A else shared.get("targets_A")
                other_correct = shared.get("correct_B") if is_A else shared.get("correct_A")
                other_score, _ = _partial(
                    other_traj.final_output or "", other_targets or [], other_correct or {}
                )
                team_score = 1.0 if other_score >= 0.99 else 0.0

            w = self.team_reward_weight
            final = (1 - w) * individual_score + w * team_score
            results[agent_id] = RewardResult(
                name="coop_letter_counting",
                score=final,
                weight=1.0,
                reason=f"{agent_id} ind={individual_score:.2f} team={team_score:.1f} {reason}",
            )

        return results

    def snapshot(self) -> dict[str, Any]:
        return self._inner.snapshot()


# ---------------------------------------------------------------------------
# DebateEnv: adversarial debate multi-agent env
# ---------------------------------------------------------------------------


class DebateEnv(MultiAgentEnv):
    """Two-agent debate environment.

    Agents: "proposer" and "skeptic".
    - Proposer: claims an answer and argues for it.
    - Skeptic: critiques the proposer's argument.
    - Judge reward: a judge function evaluates the final debate transcript.

    This environment requires an external judge (ORM/LLM-as-judge) to be
    injected as ``judge_fn: async (item, proposer_text, skeptic_text) → float``.
    """

    def __init__(
        self,
        inner_env: BaseEnv,
        judge_fn: Any,  # async callable
        debate_rounds: int = 1,
    ) -> None:
        self._inner = inner_env
        self._judge = judge_fn
        self.debate_rounds = debate_rounds

    def get_agents(self) -> list[str]:
        return ["proposer", "skeptic"]

    async def setup(self) -> None:
        await self._inner.setup()

    async def close(self) -> None:
        close_fn = getattr(self._inner, "close", None)
        if close_fn is not None:
            await close_fn()

    async def get_next_multi_item(self) -> MultiAgentItem:
        item = await self._inner.get_next_item()
        question = self._inner.format_prompt(item)
        return MultiAgentItem(
            base_item=item,
            agent_observations={
                "proposer": (
                    f"You are the PROPOSER. Provide an answer and argue for it.\n\n{question}"
                ),
                "skeptic": (
                    f"You are the SKEPTIC. Critique the proposer's answer. "
                    f"You will see the proposer's response on the message board.\n\n{question}"
                ),
            },
        )

    async def compute_multi_reward(
        self,
        item: MultiAgentItem,
        trajectories: dict[str, Trajectory],
        board: MessageBoard,
        tool_context: Any,
    ) -> dict[str, RewardResult]:
        proposer_text = (
            trajectories.get("proposer") or _empty_trajectory()
        ).final_output or ""
        skeptic_text = (
            trajectories.get("skeptic") or _empty_trajectory()
        ).final_output or ""

        try:
            judge_score = float(await self._judge(item.base_item, proposer_text, skeptic_text))
        except Exception:
            judge_score = 0.0

        # Proposer gets the judge score; Skeptic gets 1 - judge_score
        # (incentivizes skeptic to find real flaws)
        return {
            "proposer": RewardResult(
                name="debate_proposer",
                score=judge_score,
                weight=1.0,
                reason=f"judge={judge_score:.3f}",
            ),
            "skeptic": RewardResult(
                name="debate_skeptic",
                score=1.0 - judge_score,
                weight=1.0,
                reason=f"adversarial judge={1 - judge_score:.3f}",
            ),
        }
