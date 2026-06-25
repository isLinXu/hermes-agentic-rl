"""Curriculum Scheduler for Agentic RL Training
==============================================

Implements explicit curriculum stages with progressive difficulty,
addressing the core issue: the model learns to "game" rewards rather
than develop genuine tool-use competence.

Design
------
The curriculum has three stages, each with specific learning objectives:

1. **Tool Call Structure** (Stage 1):
   Reward valid tool call JSON format. The model learns *when* to emit
   a tool call and *how* to format it correctly.

2. **Tool Call Content** (Stage 2):
   Reward correct tool call content (right tool name, valid arguments).
   The model learns *which* tool to call and *what* arguments to pass.

3. **Summary & Reasoning Quality** (Stage 3):
   Reward concise, informative summaries and correct reasoning chains.
   The model learns to synthesize tool results into coherent answers.

Progression Logic
------------------
A stage is considered "mastered" when the rolling mean of its primary
metric exceeds a configurable threshold for a sustained number of
evaluations. This prevents premature advancement due to noise.

Integration
-----------
The scheduler integrates with:
  - ``DynamicRewardBalancer`` for per-component weight scheduling
  - ``RewardComposer`` for conditional activation of curriculum-gated rewards
  - ``shaping.curriculum_shaping`` for curriculum-aware reward shaping
  - ``OnPolicyTrainer`` via the ``multi_turn_credit`` config
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.rewards.toolcall_reward import score_tool_calls

# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CurriculumStage:
    """A single stage in the curriculum.

    Attributes:
        name: Human-readable stage name.
        reward_weight: Base weight for this stage's reward component.
        mastery_threshold: Rolling mean score above which the stage is
            considered mastered (range [0, 1]).
        sustain_evals: Number of consecutive evaluations above threshold
            required before advancing.
        primary_metric: Key in batch_stats to track for mastery.
    """

    name: str
    reward_weight: float = 1.0
    mastery_threshold: float = 0.7
    sustain_evals: int = 5
    primary_metric: str = ""

    # Internal state (not serialized by default)
    _rolling_scores: list[float] = field(default_factory=list, repr=False)
    _consecutive_above: int = 0

    def observe(self, score: float) -> None:
        """Record a score observation for mastery tracking."""
        self._rolling_scores.append(score)
        # Keep only last 20 observations for rolling mean
        if len(self._rolling_scores) > 20:
            self._rolling_scores = self._rolling_scores[-20:]

        if score >= self.mastery_threshold:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

    def is_mastered(self) -> bool:
        """Check if this stage has been mastered."""
        if not self._rolling_scores:
            return False
        return self._consecutive_above >= self.sustain_evals

    def rolling_mean(self) -> float:
        """Compute rolling mean of recent scores."""
        if not self._rolling_scores:
            return 0.0
        return sum(self._rolling_scores) / len(self._rolling_scores)

    def reset(self) -> None:
        """Reset internal tracking state."""
        self._rolling_scores.clear()
        self._consecutive_above = 0


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CurriculumSchedulerConfig:
    """Configuration for the curriculum scheduler."""

    # Whether to auto-advance stages when mastery is achieved.
    auto_advance: bool = True

    # Minimum number of iterations before allowing stage advancement.
    min_iters_per_stage: int = 10

    # Whether to allow regression (go back to a previous stage if
    # performance drops significantly).
    allow_regression: bool = False

    # Regression threshold: if rolling mean drops below
    # mastery_threshold * regression_factor, consider regressing.
    regression_factor: float = 0.5


class CurriculumScheduler:
    """Manages curriculum progression through defined stages.

    The scheduler tracks per-stage metrics and advances through stages
    when mastery conditions are met. It integrates with the training
    loop via ``observe_batch_stats`` which is called after each
    training iteration.

    Usage::

        scheduler = CurriculumScheduler(stages=create_default_curriculum())
        for iter_idx in range(n_iters):
            # ... training step ...
            batch_stats = compute_batch_stats(...)
            scheduler.observe_batch_stats(iter_idx, batch_stats)
            if scheduler.should_advance():
                scheduler.advance()
            current_stage = scheduler.get_current_stage()
    """

    def __init__(
        self,
        stages: list[CurriculumStage],
        cfg: CurriculumSchedulerConfig | None = None,
    ) -> None:
        if not stages:
            raise ValueError("Curriculum must have at least one stage")
        self.stages = stages
        self.cfg = cfg or CurriculumSchedulerConfig()
        self._current_idx = 0
        self._iter_count = 0
        self._iters_in_stage = 0
        self._completed_stages: list[str] = []

    @property
    def current_stage_index(self) -> int:
        """Current stage index (0-based)."""
        return self._current_idx

    @property
    def completed_stages(self) -> list[str]:
        """Names of completed stages."""
        return list(self._completed_stages)

    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        if self._current_idx >= len(self.stages):
            return self.stages[-1]
        return self.stages[self._current_idx]

    def is_complete(self) -> bool:
        """Check if all curriculum stages are completed."""
        return self._current_idx >= len(self.stages)

    def observe_batch_stats(
        self,
        iter_idx: int,
        batch_stats: dict[str, Any],
    ) -> None:
        """Update stage tracking from batch statistics.

        Extracts the primary metric for the current stage from
        batch_stats and records it for mastery tracking.

        Args:
            iter_idx: Current training iteration.
            batch_stats: Dict with training metrics.
        """
        self._iter_count = iter_idx
        self._iters_in_stage += 1

        if self.is_complete():
            return

        stage = self.get_current_stage()
        metric_key = stage.primary_metric
        if not metric_key:
            return

        # Try to extract the metric from batch_stats
        score = _extract_metric(batch_stats, metric_key)
        if score is not None:
            stage.observe(score)

    def should_advance(self) -> bool:
        """Check whether the current stage should advance.

        Returns True if:
        1. Auto-advance is enabled
        2. Minimum iterations per stage have been met
        3. Current stage is mastered
        """
        if not self.cfg.auto_advance:
            return False
        if self.is_complete():
            return False
        if self._iters_in_stage < self.cfg.min_iters_per_stage:
            return False
        return self.get_current_stage().is_mastered()

    def advance(self) -> CurriculumStage | None:
        """Advance to the next curriculum stage.

        Returns:
            The new current stage, or None if curriculum is complete.
        """
        if self.is_complete():
            return None

        old_stage = self.get_current_stage()
        self._completed_stages.append(old_stage.name)
        self._current_idx += 1
        self._iters_in_stage = 0

        if self.is_complete():
            return None
        return self.get_current_stage()

    def get_stage_weights(self) -> dict[str, float]:
        """Get reward weights for all stages.

        Current stage gets its full weight. Future stages get 0.
        Completed stages get a reduced weight (decay factor 0.5)
        to maintain the learned behavior.
        """
        weights: dict[str, float] = {}
        for i, stage in enumerate(self.stages):
            if i < self._current_idx:
                # Completed stage: reduced weight to maintain behavior
                weights[stage.name] = stage.reward_weight * 0.5
            elif i == self._current_idx:
                # Current stage: full weight
                weights[stage.name] = stage.reward_weight
            else:
                # Future stage: not yet active
                weights[stage.name] = 0.0
        return weights

    def snapshot(self) -> dict[str, Any]:
        """Return a logging-friendly summary."""
        stage = self.get_current_stage()
        return {
            "curriculum/current_stage": stage.name,
            "curriculum/current_idx": self._current_idx,
            "curriculum/iter_count": self._iter_count,
            "curriculum/iters_in_stage": self._iters_in_stage,
            "curriculum/is_complete": self.is_complete(),
            "curriculum/rolling_mean": stage.rolling_mean(),
            "curriculum/consecutive_above": stage._consecutive_above,
            "curriculum/completed": list(self._completed_stages),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "current_idx": self._current_idx,
            "iter_count": self._iter_count,
            "iters_in_stage": self._iters_in_stage,
            "completed_stages": list(self._completed_stages),
            "stage_scores": [list(s._rolling_scores) for s in self.stages],
            "stage_consecutive": [s._consecutive_above for s in self.stages],
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint."""
        self._current_idx = int(d.get("current_idx", 0))
        self._iter_count = int(d.get("iter_count", 0))
        self._iters_in_stage = int(d.get("iters_in_stage", 0))
        self._completed_stages = list(d.get("completed_stages", []))
        stage_scores = d.get("stage_scores", [])
        stage_consecutive = d.get("stage_consecutive", [])
        for i, (scores, consec) in enumerate(zip(stage_scores, stage_consecutive, strict=False)):
            if i < len(self.stages):
                self.stages[i]._rolling_scores = list(scores)
                self.stages[i]._consecutive_above = int(consec)


# ---------------------------------------------------------------------------
# Default curriculum factory
# ---------------------------------------------------------------------------


def create_default_curriculum() -> list[CurriculumStage]:
    """Create the default three-stage curriculum.

    Stages:
      1. tool_call_structure: Learn to emit valid tool call format
      2. tool_call_content: Learn to call the right tool with correct args
      3. summary_quality: Learn to synthesize results into correct answers
    """
    return [
        CurriculumStage(
            name="tool_call_structure",
            reward_weight=0.5,
            mastery_threshold=0.7,
            sustain_evals=5,
            primary_metric="reward_toolcall_reward_mean_name_score",
        ),
        CurriculumStage(
            name="tool_call_content",
            reward_weight=0.7,
            mastery_threshold=0.6,
            sustain_evals=5,
            primary_metric="reward_toolcall_reward_mean_value_score",
        ),
        CurriculumStage(
            name="summary_quality",
            reward_weight=1.0,
            mastery_threshold=0.5,
            sustain_evals=5,
            primary_metric="reward_outcome_mean",
        ),
    ]


# ---------------------------------------------------------------------------
# Turn-level reward computation
# ---------------------------------------------------------------------------


def compute_turn_reward(
    response: str,
    expected_tool_names: list[str] | None = None,
    require_think_block: bool = False,
) -> tuple[float, bool]:
    """Compute reward for a single turn in a multi-turn trajectory.

    This function:
    1. Extracts tool calls from the response (JSON parsing)
    2. Validates them against expected tool names
    3. Computes a reward based on tool call quality

    Args:
        response: The model's response text for this turn.
        expected_tool_names: Optional list of expected tool names.
        require_think_block: Whether to require a <think> block.

    Returns:
        (reward, is_valid_turn) where:
        - reward: Computed reward value in [0.0, 1.0]
        - is_valid_turn: Whether the turn passed validation
    """
    # Try to extract tool calls from the response
    tool_calls = _extract_tool_calls_from_text(response)

    if not tool_calls:
        # No tool calls found — could be a summary/narration turn
        # Check if it's a valid non-tool-call response
        is_valid_summary = len(response.strip()) > 10
        return (0.3 if is_valid_summary else 0.0, is_valid_summary)

    # Score tool calls using the existing ToolcallReward infrastructure
    summary = score_tool_calls(tool_calls)
    reward = float(summary["score"])
    is_valid = summary["invalid_calls"] == 0

    # Bonus for matching expected tool names
    if expected_tool_names:
        matched_names = sum(
            1
            for call in tool_calls
            if isinstance(call, dict) and _extract_call_name(call) in expected_tool_names
        )
        name_bonus = matched_names / max(1, len(expected_tool_names))
        reward = 0.6 * reward + 0.4 * name_bonus

    return (reward, is_valid)


def compute_outcome_reward(
    response: str,
    gold_answer: str,
) -> tuple[float, bool]:
    """Compute outcome reward for the entire episode.

    Uses the token_level_reward module for robust answer comparison
    including partial credit and boxed answer extraction.

    Args:
        response: Model's final response.
        gold_answer: Expected answer.

    Returns:
        (reward, is_correct) tuple.
    """
    from hermes_agentic_rl.rewards.token_level_reward import compute_outcome_reward

    reward = compute_outcome_reward(response, gold_answer, partial_credit=True)
    return (reward, reward >= 0.9)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_metric(batch_stats: dict[str, Any], key: str) -> float | None:
    """Extract a metric value from batch_stats using dot-notation key.

    Supports both flat keys ("reward_outcome_mean") and nested
    keys ("reward_components.outcome.mean").
    """
    # Try flat key first
    if key in batch_stats:
        val = batch_stats[key]
        if isinstance(val, int | float):
            return float(val)

    # Try nested key with dot notation
    parts = key.split(".")
    current: Any = batch_stats
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None

    if isinstance(current, int | float):
        return float(current)
    return None


def _extract_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool call JSON objects from response text.

    Looks for patterns like:
    - ```json ... ``` blocks
    - <tool_call>...</tool_call> tags
    - Direct JSON objects with "name" or "function" keys
    """
    import re

    calls: list[dict[str, Any]] = []

    # Pattern 1: ```json ... ``` code blocks
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                calls.append(parsed)
            elif isinstance(parsed, list):
                calls.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            pass

    # Pattern 2: <tool_call>...</tool_call> tags
    tool_call_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    for block in tool_call_blocks:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                calls.append(parsed)
        except json.JSONDecodeError:
            pass

    # Pattern 3: Standalone JSON objects with tool call structure
    # Look for {"name": ...} or {"function": {"name": ...}} patterns
    json_objects = re.findall(r"\{[^{}]*\}", text)
    for obj_str in json_objects:
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and ("name" in parsed or "function" in parsed):
                calls.append(parsed)
        except json.JSONDecodeError:
            pass

    return calls


def _extract_call_name(call: dict[str, Any]) -> str | None:
    """Extract tool name from a call dict."""
    name = call.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    func = call.get("function")
    if isinstance(func, dict):
        func_name = func.get("name")
        if isinstance(func_name, str) and func_name.strip():
            return func_name.strip()
    return None
