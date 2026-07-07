"""RULER — Rule-based Automatic Reward Estimator.

Inspired by OpenPipe/ART's RULER system, this module provides a declarative
rule engine that generates reward signals from trajectory metadata without
requiring hand-coded reward functions.

Design
------
A RULER *rule* is a triple (predicate, scorer, weight). Given a trajectory:

1. ``predicate(item, trajectory) -> bool`` decides if the rule fires.
2. ``scorer(item, trajectory) -> float`` produces a raw score in [0, 1].
3. ``weight`` scales the contribution in the aggregate.

Rules are registered declaratively via a YAML/dict spec or programmatically.
This bridges the gap between:

- **Hand-coded rewards** (current approach: ToolcallReward, OutcomeReward, ...)
- **Learned rewards** (PRM, LLM-as-judge — expensive, slow)

RULER fills the middle ground: fast, interpretable, auto-generated from task
metadata templates.

Built-in rule templates:
    - exact_match: Score 1.0 if final_output matches gold_answer.
    - regex_match: Score 1.0 if final_output matches a regex pattern.
    - tool_call_count: Score = min(n_calls / expected, 1.0).
    - turn_efficiency: Score = 1.0 - min(turns_used / max_turns, 0.9).
    - format_valid: Score 1.0 if output parses as valid JSON/XML.
    - length_penalty: Score based on response length vs. target range.

Usage::

    from hermes_agentic_rl.rewards.ruler import RULER, RULERConfig

    ruler = RULER.from_config({
        "rules": [
            {"name": "exact_match", "template": "exact_match", "weight": 2.0,
             "params": {"gold_key": "answer"}},
            {"name": "tool_usage", "template": "tool_call_count", "weight": 1.0,
             "params": {"expected": 3}},
        ],
    })

    # Use as a BaseReward component in RewardComposer:
    composer = RewardComposer(components=[ruler], config=...)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward

logger = logging.getLogger(__name__)

# Type aliases
PredicateFn = Callable[[dict[str, Any], Trajectory], bool]
ScorerFn = Callable[[dict[str, Any], Trajectory], float]


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RULERRule:
    """A single RULER rule.

    Attributes
    ----------
    name : str
        Human-readable rule name (used as component name in RewardResult).
    template : str
        Built-in template name or "custom".
    predicate : PredicateFn
        Returns True if this rule should fire for the given item/trajectory.
    scorer : ScorerFn
        Produces a raw score in [0, 1].
    weight : float
        Weight in the aggregate reward.
    params : dict
        Template parameters (for logging/debugging).
    """

    name: str
    template: str
    predicate: PredicateFn
    scorer: ScorerFn
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in rule templates
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: dict[str, tuple[PredicateFn, ScorerFn]] = {}


def _register(template_name: str) -> Callable:
    """Decorator to register a built-in rule template."""
    def decorator(func: Callable) -> Callable:
        def wrapper(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
            return func(params)
        _BUILTIN_TEMPLATES[template_name] = wrapper
        return wrapper
    return decorator


def _always_true(_item: dict[str, Any], _traj: Trajectory) -> bool:
    return True


# --- exact_match ---

@_register("exact_match")
def _template_exact_match(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
    """Score 1.0 if final_output matches the gold answer."""
    gold_key = params.get("gold_key", "answer")
    normalize = params.get("normalize", True)

    def predicate(item: dict[str, Any], traj: Trajectory) -> bool:
        return gold_key in item and traj.final_output is not None

    def scorer(item: dict[str, Any], traj: Trajectory) -> float:
        gold = str(item.get(gold_key, "")).strip()
        actual = str(traj.final_output or "").strip()
        if normalize:
            gold = gold.lower()
            actual = actual.lower()
        return 1.0 if gold and actual and gold == actual else 0.0

    return predicate, scorer


# --- regex_match ---

@_register("regex_match")
def _template_regex_match(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
    """Score 1.0 if final_output matches a regex pattern."""
    pattern = params.get("pattern", "")
    compiled = re.compile(pattern) if pattern else None

    def predicate(_item: dict[str, Any], traj: Trajectory) -> bool:
        return compiled is not None and traj.final_output is not None

    def scorer(_item: dict[str, Any], traj: Trajectory) -> float:
        if compiled is None or not traj.final_output:
            return 0.0
        return 1.0 if compiled.search(traj.final_output) else 0.0

    return predicate, scorer


# --- tool_call_count ---

@_register("tool_call_count")
def _template_tool_call_count(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
    """Score = min(n_tool_calls / expected, 1.0)."""
    expected = max(1, int(params.get("expected", 1)))

    def predicate(_item: dict[str, Any], traj: Trajectory) -> bool:
        return len(traj.steps) > 0

    def scorer(_item: dict[str, Any], traj: Trajectory) -> float:
        n_calls = sum(
            1 for step in traj.steps
            if getattr(step, "tool_call", None) is not None
        )
        return min(n_calls / expected, 1.0)

    return predicate, scorer


# --- turn_efficiency ---

@_register("turn_efficiency")
def _template_turn_efficiency(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
    """Score = 1.0 - min(turns_used / max_turns, 0.9)."""
    max_turns = max(1, int(params.get("max_turns", 10)))

    def predicate(_item: dict[str, Any], traj: Trajectory) -> bool:
        return traj.turns_used > 0

    def scorer(_item: dict[str, Any], traj: Trajectory) -> float:
        return 1.0 - min(traj.turns_used / max_turns, 0.9)

    return predicate, scorer


# --- format_valid ---

@_register("format_valid")
def _template_format_valid(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
    """Score 1.0 if output parses as valid JSON or XML."""
    fmt = params.get("format", "json")

    def predicate(_item: dict[str, Any], traj: Trajectory) -> bool:
        return traj.final_output is not None

    def scorer(_item: dict[str, Any], traj: Trajectory) -> float:
        output = (traj.final_output or "").strip()
        if not output:
            return 0.0
        if fmt == "json":
            try:
                json.loads(output)
                return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0
        elif fmt == "xml":
            # Simple check: balanced tags
            return 1.0 if output.startswith("<") and output.endswith(">") else 0.0
        return 0.0

    return predicate, scorer


# --- length_penalty ---

@_register("length_penalty")
def _template_length_penalty(params: dict[str, Any]) -> tuple[PredicateFn, ScorerFn]:
    """Score based on response length vs. target range."""
    min_len = int(params.get("min_len", 10))
    max_len = int(params.get("max_len", 500))
    target = int(params.get("target", (min_len + max_len) // 2))

    def predicate(_item: dict[str, Any], traj: Trajectory) -> bool:
        return traj.final_output is not None

    def scorer(_item: dict[str, Any], traj: Trajectory) -> float:
        length = len((traj.final_output or "").strip())
        if length == 0:
            return 0.0
        if length < min_len or length > max_len:
            return 0.2
        # Gaussian-like peak around target
        deviation = abs(length - target) / max(1, max_len - min_len)
        return max(0.0, 1.0 - deviation)

    return predicate, scorer


# ---------------------------------------------------------------------------
# RULER config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RULERConfig:
    """Declarative configuration for RULER.

    Attributes
    ----------
    rules : list[dict]
        List of rule specs. Each spec has:
        - name: str (rule name)
        - template: str (built-in template or "custom")
        - weight: float (default 1.0)
        - params: dict (template-specific parameters)
        - predicate: optional callable (for custom rules)
        - scorer: optional callable (for custom rules)
    default_weight : float
        Default weight for rules that don't specify one.
    """

    rules: list[dict[str, Any]] = field(default_factory=list)
    default_weight: float = 1.0


# ---------------------------------------------------------------------------
# RULER reward component
# ---------------------------------------------------------------------------


class RULER(BaseReward):
    """Rule-based Automatic Reward Estimator.

    A ``BaseReward`` implementation that evaluates a trajectory against
    a set of declarative rules. Each rule that fires contributes a
    weighted score; the final score is the weighted average of all
    fired rules.

    Can be used directly in ``RewardComposer`` or ``RewardManager``::

        composer = RewardComposer(components=[ruler, outcome_reward], ...)
    """

    name = "ruler"

    def __init__(
        self,
        rules: list[RULERRule] | None = None,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rules: list[RULERRule] = rules or []
        self.weight = weight

    @classmethod
    def from_config(cls, config: dict[str, Any] | RULERConfig) -> RULER:
        """Build a RULER from a declarative config.

        Parameters
        ----------
        config : dict or RULERConfig
            Configuration with a ``rules`` key containing a list of rule specs.

        Returns
        -------
        RULER
            Configured RULER instance.
        """
        if isinstance(config, dict):
            config = RULERConfig(
                rules=config.get("rules", []),
                default_weight=config.get("default_weight", 1.0),
            )

        rules: list[RULERRule] = []
        for spec in config.rules:
            template = spec.get("template", "custom")
            params = spec.get("params", {})

            if template == "custom":
                predicate = spec.get("predicate", _always_true)
                scorer_fn = spec.get("scorer", lambda *_: 0.0)
            elif template in _BUILTIN_TEMPLATES:
                predicate, scorer_fn = _BUILTIN_TEMPLATES[template](params)
            else:
                logger.warning(
                    f"RULER: unknown template {template!r} for rule "
                    f"{spec.get('name', 'unnamed')}; skipping"
                )
                continue

            rules.append(RULERRule(
                name=spec.get("name", f"ruler_{template}"),
                template=template,
                predicate=predicate,
                scorer=scorer_fn,
                weight=spec.get("weight", config.default_weight),
                params=params,
            ))

        return cls(rules=rules, weight=config.default_weight)

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        """Evaluate all rules against the trajectory."""
        fired_scores: list[tuple[str, float, float]] = []  # (name, score, weight)
        skipped: list[str] = []

        for rule in self._rules:
            try:
                if rule.predicate(item, trajectory):
                    raw_score = max(0.0, min(1.0, float(rule.scorer(item, trajectory))))
                    fired_scores.append((rule.name, raw_score, rule.weight))
                else:
                    skipped.append(rule.name)
            except Exception as exc:
                logger.warning(
                    f"RULER rule {rule.name!r} raised: {type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                skipped.append(rule.name)

        if not fired_scores:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="No rules fired",
                weight=self.weight,
                metadata={
                    "fired_rules": [],
                    "skipped_rules": skipped,
                    "total_rules": len(self._rules),
                },
            )

        total_weight = sum(w for _, _, w in fired_scores)
        if total_weight <= 0:
            final_score = 0.0
        else:
            final_score = sum(s * w for _, s, w in fired_scores) / total_weight

        return RewardResult(
            name=self.name,
            score=final_score,
            reason=f"{len(fired_scores)}/{len(self._rules)} rules fired",
            weight=self.weight,
            metadata={
                "fired_rules": [
                    {"name": n, "score": s, "weight": w}
                    for n, s, w in fired_scores
                ],
                "skipped_rules": skipped,
                "total_rules": len(self._rules),
                "raw_scores": {n: s for n, s, _ in fired_scores},
            },
        )

    @property
    def rules(self) -> list[RULERRule]:
        """Read-only access to the rule list."""
        return list(self._rules)

    def add_rule(
        self,
        name: str,
        template: str,
        weight: float = 1.0,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Add a rule at runtime."""
        params = params or {}
        if template == "custom":
            raise ValueError("Use add_custom_rule for custom rules")
        if template not in _BUILTIN_TEMPLATES:
            raise ValueError(f"Unknown template: {template!r}")
        predicate, scorer = _BUILTIN_TEMPLATES[template](params)
        self._rules.append(RULERRule(
            name=name, template=template, predicate=predicate,
            scorer=scorer, weight=weight, params=params,
        ))

    def add_custom_rule(
        self,
        name: str,
        predicate: PredicateFn,
        scorer: ScorerFn,
        weight: float = 1.0,
    ) -> None:
        """Add a custom rule at runtime."""
        self._rules.append(RULERRule(
            name=name, template="custom", predicate=predicate,
            scorer=scorer, weight=weight,
        ))

    def list_templates(self) -> list[str]:
        """List available built-in templates."""
        return sorted(_BUILTIN_TEMPLATES.keys())
