"""Cross-Session Memory-Aware Reward Shaping.

Differentiator over OpenClaw-RL
--------------------------------
OpenClaw-RL scores each episode in isolation — it cannot tell whether an agent
has already solved a similar task in a previous session or whether it is
regressing on a domain it used to handle well.

``MemoryAwareRewardShaper`` maintains a lightweight rolling memory of past
rewards *per task-type / prompt-hash*. It computes an **improvement bonus**:
tasks where the agent scores significantly better than its own historical
baseline receive a bonus, reinforcing generalisation; tasks where the agent
regresses receive a penalty.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~~
For task key k and current outcome score r_new:

    memory_mean(k)  = EMA of past scores for k     (decay = alpha)
    memory_std(k)   = EMA of past std for k

    z(k) = (r_new - memory_mean(k)) / max(memory_std(k), sigma_floor)

    bonus(k) = clip(bonus_coef * z(k), -clip_max, clip_max)

    r_shaped = r_new + bonus(k)

When the task key is new (no history), no bonus/penalty is applied (bonus = 0).

Key properties
~~~~~~~~~~~~~~
- Pure Python / no external dependencies beyond the core package.
- Thread-safe: ``MemoryAwareRewardShaper.update`` can be called from the
  trainer's main thread with no locking needed (single-threaded training).
- Configurable per-axis: ``axis_bonus_coef`` overrides the global
  ``bonus_coef`` for specific capability axes, enabling targeted improvement
  incentives (e.g. emphasise tool-use progress without over-rewarding
  reasoning regressions).
- Efficient: all state is stored in two dicts of EMA scalars — O(1) per call.

Usage in trainer::

    shaper = MemoryAwareRewardShaper(MemoryRewardConfig())
    # After reward evaluation, before advantage computation:
    records = shaper.shape_records(records, tokenizer=policy.tokenizer)

    # At training end / checkpoint time:
    state = shaper.state_dict()
    # ...
    shaper.load_state_dict(state)   # restore across sessions
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.algos.base import RolloutRecord

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryRewardConfig:
    """Configuration for the cross-session memory-aware reward shaper."""

    # EMA decay: closer to 1.0 → longer memory (slower adaptation).
    alpha: float = 0.9

    # Coefficient applied to the z-score to produce the bonus.
    bonus_coef: float = 0.1

    # Max absolute bonus to avoid destabilising the reward scale.
    clip_max: float = 0.5

    # Floor on standard deviation to prevent division by near-zero.
    sigma_floor: float = 0.1

    # Minimum number of observations before a bonus is applied.
    # (Avoids spurious bonuses from a single data point.)
    min_obs: int = 3

    # Per-axis overrides: {axis_name: bonus_coef}.
    # Axis is read from record.metadata["opd_hint_axis"] (set by OPD teacher).
    axis_bonus_coef: dict[str, float] = field(default_factory=dict)

    # How to build the task key. Options:
    #   "task_id"      — use record.group_id or metadata["task_id"] (default)
    #   "prompt_hash"  — SHA-1 of the first 128 chars of prompt token IDs
    task_key_mode: str = "task_id"

    # When True, apply the shaping to the record.reward in-place AND store
    # the original under metadata["pre_shaping_reward"].
    apply_inplace: bool = True


# ---------------------------------------------------------------------------
# EMA state
# ---------------------------------------------------------------------------


class _EMAState:
    """Single exponential moving average with count."""

    __slots__ = ("mean", "n", "var")

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.var: float = 0.0
        self.n: int = 0

    def update(self, x: float, alpha: float) -> None:
        if self.n == 0:
            self.mean = x
            self.var = 0.0
        else:
            delta = x - self.mean
            self.mean = alpha * self.mean + (1.0 - alpha) * x
            self.var = alpha * (self.var + (1.0 - alpha) * delta * delta)
        self.n += 1

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.var))

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _EMAState:
        s = cls()
        s.mean = float(d.get("mean", 0.0))
        s.var = float(d.get("var", 0.0))
        s.n = int(d.get("n", 0))
        return s


# ---------------------------------------------------------------------------
# Shaper
# ---------------------------------------------------------------------------


class MemoryAwareRewardShaper:
    """Cross-session memory-aware reward shaper.

    Maintains rolling per-task-key EMA statistics and adds an improvement
    bonus (or regression penalty) to each record's reward.
    """

    def __init__(self, cfg: MemoryRewardConfig | None = None) -> None:
        self.cfg = cfg or MemoryRewardConfig()
        self._memory: dict[str, _EMAState] = {}
        self._total_shaped = 0
        self._total_bonus = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def shape_records(
        self,
        records: list[RolloutRecord],
        *,
        tokenizer: Any | None = None,
    ) -> list[RolloutRecord]:
        """Apply memory-aware shaping to a list of records.

        Returns the (optionally mutated) records with shaped rewards.
        """
        cfg = self.cfg
        for rec in records:
            key = self._task_key(rec, tokenizer=tokenizer)
            state = self._memory.get(key)

            # Compute bonus only when we have enough history.
            if state is not None and state.n >= cfg.min_obs:
                axis = rec.metadata.get("opd_hint_axis")
                coef = (
                    float(cfg.axis_bonus_coef.get(str(axis), cfg.bonus_coef))
                    if axis is not None
                    else cfg.bonus_coef
                )
                sigma = max(state.std, cfg.sigma_floor)
                z = (float(rec.reward) - state.mean) / sigma
                bonus = float(max(-cfg.clip_max, min(cfg.clip_max, coef * z)))
            else:
                bonus = 0.0

            # Update memory with the current score AFTER computing bonus
            # (so the bonus is relative to the mean BEFORE this episode).
            if key not in self._memory:
                self._memory[key] = _EMAState()
            self._memory[key].update(float(rec.reward), cfg.alpha)

            if bonus != 0.0 and cfg.apply_inplace:
                if cfg.apply_inplace:
                    rec.metadata["pre_shaping_reward"] = float(rec.reward)
                    rec.metadata["memory_shaping_bonus"] = bonus
                    rec.metadata["memory_task_key"] = key
                    rec.reward = float(rec.reward) + bonus
                self._total_shaped += 1
                self._total_bonus += bonus

        return records

    def update_from_record(
        self,
        record: RolloutRecord,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        """Update memory from a single record without shaping."""
        key = self._task_key(record, tokenizer=tokenizer)
        if key not in self._memory:
            self._memory[key] = _EMAState()
        self._memory[key].update(float(record.reward), self.cfg.alpha)

    def get_task_stats(self, key: str) -> dict[str, Any] | None:
        """Return EMA stats for a task key, or None if unseen."""
        state = self._memory.get(key)
        if state is None:
            return None
        return {"mean": state.mean, "std": state.std, "n": state.n}

    def snapshot(self) -> dict[str, Any]:
        """Return summary stats for logging."""
        return {
            "memory_n_keys": len(self._memory),
            "memory_total_shaped": float(self._total_shaped),
            "memory_total_bonus": self._total_bonus,
        }

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "memory": {k: v.to_dict() for k, v in self._memory.items()},
            "total_shaped": self._total_shaped,
            "total_bonus": self._total_bonus,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        mem = d.get("memory", {})
        self._memory = {k: _EMAState.from_dict(v) for k, v in mem.items()}
        self._total_shaped = int(d.get("total_shaped", 0))
        self._total_bonus = float(d.get("total_bonus", 0.0))

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _task_key(
        self,
        rec: RolloutRecord,
        *,
        tokenizer: Any | None = None,
    ) -> str:
        mode = self.cfg.task_key_mode
        if mode == "prompt_hash":
            # SHA-1 of the first 128 token IDs as a compact, collision-resistant key.
            token_prefix = tuple(rec.prompt_ids[:128])
            h = hashlib.sha1(str(token_prefix).encode()).hexdigest()[:16]
            return f"ph_{h}"
        else:  # "task_id" (default)
            tid = rec.metadata.get("task_id") or rec.group_id or ""
            # Strip the ::turn:N suffix so all turns of one episode share a key.
            if "::turn:" in str(tid):
                tid = str(tid).split("::turn:")[0]
            return f"tid_{tid}" if tid else "tid_unknown"
