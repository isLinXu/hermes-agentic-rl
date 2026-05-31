"""Next-State Process Reward Model — from OpenClaw-RL.

Paper: OpenClaw-RL §3.1 "Infrastructure: Next-State Signal Extraction"

Key insight (§2):
  Every agent interaction generates a *next-state* signal: the user reply,
  tool output, terminal stdout/stderr, test verdict, or GUI state change that
  follows each action.  This next-state is a natural evaluative signal —
  no human labelling needed.

OpenClaw-RL extracts TWO complementary training signals from the next state:
  1. **Evaluative** (scalar r ∈ {+1, −1, 0}): how well did the action
     perform?  Computed via majority vote across m independent PRM calls.
  2. **Directive** (textual hint): how should the action have been different?
     Extracted when the PRM judges the next-state as corrective (used by OPD).

This module implements the evaluative path.

NextStatePRM protocol:
  • Each judge call receives (response, next_state) and returns a vote
    (GOOD=+1, BAD=−1, NEUTRAL=0) plus an optional hint string.
  • m independent calls are made asynchronously.
  • Final reward = majority vote: +1 if ≥⌈m/2⌉ are GOOD, −1 if ≥⌈m/2⌉ BAD,
    0 (NEUTRAL) otherwise.
  • At-least-one guarantee: the last turn in a session is always scored even
    if no next-state arrives yet (score = 0 with loss_mask=1).

Integration with hermes-agentic-rl RewardManager:
  Use NextStatePRMComponent as a drop-in BaseReward replacement.
  It reads trajectory.metadata["runtime"]["next_state"] for the next-state
  signal and calls the judge asynchronously.

Integration with OPD:
  When a GOOD vote is accompanied by a hint, the component writes it to
  trajectory.metadata["runtime"]["rl"]["opd_hint"] so the trainer can
  build the teacher distribution for the OPD branch.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory

# ---------------------------------------------------------------------------
# Vote types
# ---------------------------------------------------------------------------

GOOD    = +1
BAD     = -1
NEUTRAL = 0

VOTE_LABELS = {GOOD: "GOOD", BAD: "BAD", NEUTRAL: "NEUTRAL"}


@dataclass(slots=True)
class PRMVote:
    """Result of one judge call."""
    vote:  int          # GOOD / BAD / NEUTRAL
    hint:  str | None   # optional directive hint
    raw:   str = ""     # raw judge output (for debugging)


# ---------------------------------------------------------------------------
# Judge interface
# ---------------------------------------------------------------------------


class NextStateJudge:
    """Abstract judge: given (response, next_state) produce a PRMVote.

    Subclass or inject a ``judge_fn`` callable:
        async judge_fn(response: str, next_state: str) -> PRMVote

    The default parse logic looks for simple patterns in the judge output:
        "GOOD" / "CORRECT" / "+1"  → GOOD
        "BAD"  / "WRONG"   / "-1"  → BAD
        anything else              → NEUTRAL

    Optionally extracts a [HINT_START]...[HINT_END] block from the output.
    """

    def __init__(self, judge_fn: Any) -> None:
        self._fn = judge_fn

    async def judge(self, response: str, next_state: str) -> PRMVote:
        raw = await self._fn(response, next_state)
        if isinstance(raw, PRMVote):
            return raw
        return self._parse(str(raw))

    @staticmethod
    def _parse(raw: str) -> PRMVote:
        import re

        text = raw.strip().upper()
        first_line = text.splitlines()[0] if text else ""
        first_token = first_line.strip().split(maxsplit=1)[0] if first_line.strip() else ""
        first_token = first_token.strip(" \t\r\n:;,.")
        if first_token in {"GOOD", "+1", "PASS", "YES"}:
            vote = GOOD
        elif first_token in {"BAD", "-1", "FAIL", "NO", "WRONG"}:
            vote = BAD
        elif first_token == "NEUTRAL":
            vote = NEUTRAL
        elif re.search(r"\b(BAD|WRONG|INCORRECT|FAIL|FAILED|NO)\b|(?<!\w)-1(?!\w)", text):
            vote = BAD
        elif re.search(r"\b(GOOD|CORRECT|PASS|YES)\b|(?<!\w)\+1(?!\w)", text):
            vote = GOOD
        else:
            vote = NEUTRAL

        m = re.search(r"\[HINT_START\](.*?)\[HINT_END\]", raw, re.DOTALL)
        hint = m.group(1).strip() if m else None
        return PRMVote(vote=vote, hint=hint, raw=raw)


# ---------------------------------------------------------------------------
# Majority vote scorer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NextStatePRMConfig:
    """Config for NextStatePRM.

    m: number of independent judge calls per (response, next_state) pair.
       OpenClaw-RL default = 3.
    at_least_one: if True (default), the last turn of a session (no next-state)
       gets score 0 with loss_mask=1 instead of being skipped entirely.
    hint_min_length: minimum character count for an extracted hint to be
       kept (avoids empty/trivial hints polluting the OPD branch).
    write_hint_to_metadata: if True, the best hint is written to
       trajectory.metadata["runtime"]["rl"]["opd_hint"].
    """
    m:                    int  = 3
    at_least_one:         bool = True
    hint_min_length:      int  = 10
    write_hint_to_metadata: bool = True


class NextStatePRM:
    """PRM judge with m-vote majority voting (OpenClaw-RL §3.1).

    Usage::

        judge = NextStateJudge(my_llm_judge_fn)
        prm   = NextStatePRM(judge, NextStatePRMConfig(m=3))

        score, hint = await prm.score(response, next_state)
        # score ∈ {+1, -1, 0}
    """

    def __init__(
        self,
        judge: NextStateJudge,
        cfg: NextStatePRMConfig | None = None,
    ) -> None:
        self.judge = judge
        self.cfg = cfg or NextStatePRMConfig()

    async def score(
        self,
        response: str,
        next_state: str | None,
    ) -> tuple[int, str | None]:
        """Return majority-voted score and the best hint.

        When next_state is None (last turn), returns (0, None) if
        at_least_one=True, else raises ValueError.
        """
        cfg = self.cfg
        if next_state is None:
            if cfg.at_least_one:
                return NEUTRAL, None
            raise ValueError("next_state is None and at_least_one=False")

        # m parallel judge calls
        tasks = [self.judge.judge(response, next_state) for _ in range(cfg.m)]
        votes: list[PRMVote] = await asyncio.gather(*tasks)

        good_count = sum(1 for v in votes if v.vote == GOOD)
        bad_count  = sum(1 for v in votes if v.vote == BAD)
        threshold  = math.ceil(cfg.m / 2)

        if good_count >= threshold:
            majority = GOOD
        elif bad_count >= threshold:
            majority = BAD
        else:
            majority = NEUTRAL

        # Select the longest useful hint from GOOD votes
        best_hint: str | None = None
        for v in votes:
            if v.vote == GOOD and v.hint and len(v.hint) >= cfg.hint_min_length:
                if best_hint is None or len(v.hint) > len(best_hint):
                    best_hint = v.hint

        return majority, best_hint


# ---------------------------------------------------------------------------
# RewardManager component
# ---------------------------------------------------------------------------


class NextStatePRMComponent:
    """Plug NextStatePRM into RewardManager as an online process reward.

    This component reads ``trajectory.metadata["runtime"]["next_state"]``
    (a string representing the environment's next-turn feedback) and calls
    the PRM judge to produce a scalar reward ∈ {+1, −1, 0}.

    It also writes the best hint to metadata for the OPD branch:
        trajectory.metadata["runtime"]["rl"]["opd_hint"]
        trajectory.metadata["runtime"]["rl"]["teacher_logprobs"]  ← filled by trainer
    """

    name = "next_state_prm"

    def __init__(
        self,
        prm: NextStatePRM,
        weight: float = 1.0,
    ) -> None:
        self.prm = prm
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context

        runtime = trajectory.metadata.get("runtime") or {}
        if not isinstance(runtime, dict):
            return RewardResult(
                name=self.name, score=0.0, weight=self.weight, reason="no runtime"
            )

        response   = trajectory.final_output or ""
        next_state = runtime.get("next_state")  # may be None for last turn

        score, hint = await self.prm.score(response, next_state)

        # Write hint to metadata for OPD branch
        if hint and self.prm.cfg.write_hint_to_metadata:
            runtime.setdefault("rl", {})
            if isinstance(runtime["rl"], dict):
                runtime["rl"]["opd_hint"] = hint
                # teacher_logprobs will be filled later by the teacher model
                # query; initialize as empty list so OPD branch can detect
                # that a hint is available.
                runtime["rl"].setdefault("teacher_logprobs", [])

        vote_label = VOTE_LABELS.get(score, "NEUTRAL")
        return RewardResult(
            name=self.name,
            score=float(score) * self.weight,
            weight=self.weight,
            reason=f"majority_vote={vote_label} hint={'yes' if hint else 'no'}",
        )
