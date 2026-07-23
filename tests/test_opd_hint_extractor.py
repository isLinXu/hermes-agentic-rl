"""Tests for the in-trainer OPD hint extractor (A2: fixes the OPD dummy-fire).

Covers:
  * Rule-based extraction from a next-state signal stamps ``opd_hint``.
  * Records that already carry a hint are left untouched (gap-filling only).
  * Records without a next-state signal are skipped.
  * Quality filter rejects too-short and low-information hints.
  * Judge errors are fail-soft (never crash; counted in stats).
  * Async judge functions are supported.
  * ``build_opd_hint_extractor`` factory honours ``type`` / disabled.
  * Full closed loop through OnPolicyTrainer: a reward that emits ONLY a
    next-state signal (no hint) still drives the OPD branch because the
    in-trainer extractor recovers the hint and the teacher filler fills it.
"""

from __future__ import annotations

from typing import Any

from hermes_agentic_rl.algos.base import RolloutRecord
from hermes_agentic_rl.rewards.opd_hint_extractor import (
    OPDHintExtractor,
    OPDHintExtractorConfig,
    build_opd_hint_extractor,
)


def _rec(
    *,
    next_state: str | None = None,
    opd_hint: str | None = None,
    final_output: str = "<answer>3</answer>",
) -> RolloutRecord:
    meta: dict[str, Any] = {"final_output": final_output}
    if next_state is not None:
        meta["next_state"] = next_state
    if opd_hint is not None:
        meta["opd_hint"] = opd_hint
    return RolloutRecord(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        old_logprobs=[0.0, 0.0],
        reward=0.0,
        group_id="g",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Rule-based extraction
# ---------------------------------------------------------------------------


def test_rule_extractor_fills_hint_from_next_state():
    ext = build_opd_hint_extractor({"type": "rule", "min_hint_chars": 5})
    assert ext is not None
    recs = [_rec(next_state="Wrong answer; correct answer is 7")]
    stats = ext.extract(recs)
    assert stats.n_hints_added == 1
    assert "<answer>7</answer>" in recs[0].metadata["opd_hint"]
    assert recs[0].metadata.get("opd_hint_source") == "in_trainer_judge"
    assert stats.as_dict()["opd_hint_effective_rate"] == 1.0


def test_extractor_leaves_existing_hint_untouched():
    ext = build_opd_hint_extractor({"type": "rule"})
    assert ext is not None
    recs = [_rec(next_state="correct answer is 7", opd_hint="keep me unchanged")]
    ext.extract(recs)
    assert recs[0].metadata["opd_hint"] == "keep me unchanged"


def test_extractor_overwrite_existing_when_configured():
    ext = build_opd_hint_extractor(
        {"type": "rule", "overwrite_existing": True, "min_hint_chars": 5}
    )
    assert ext is not None
    recs = [_rec(next_state="correct answer is 7", opd_hint="stale hint")]
    ext.extract(recs)
    assert "<answer>7</answer>" in recs[0].metadata["opd_hint"]


def test_extractor_skips_records_without_next_state():
    ext = build_opd_hint_extractor({"type": "rule"})
    assert ext is not None
    recs = [_rec()]
    stats = ext.extract(recs)
    assert stats.n_with_next_state == 0
    assert "opd_hint" not in recs[0].metadata


# ---------------------------------------------------------------------------
# Quality filtering & robustness (custom judge)
# ---------------------------------------------------------------------------


def test_quality_filter_rejects_short_and_low_info():
    async def judge(_resp: str, next_state: str) -> str:
        return next_state  # echo the signal as the "hint"

    ext = OPDHintExtractor(judge, OPDHintExtractorConfig(min_hint_chars=10, reject_low_info=True))
    short = _rec(next_state="hi")  # too short
    generic = _rec(next_state="be more helpful")  # low-info
    good = _rec(next_state="use a valid json tool argument with key 'path'")
    stats = ext.extract([short, generic, good])
    assert "opd_hint" not in short.metadata
    assert "opd_hint" not in generic.metadata
    assert good.metadata.get("opd_hint")
    assert stats.n_rejected_quality == 2
    assert stats.n_hints_added == 1


def test_judge_errors_are_fail_soft():
    def boom(_resp: str, _next_state: str) -> str:
        raise RuntimeError("judge exploded")

    ext = OPDHintExtractor(boom, OPDHintExtractorConfig())
    rec = _rec(next_state="something happened")
    stats = ext.extract([rec])  # must not raise
    assert stats.n_errors == 1
    assert "opd_hint" not in rec.metadata


def test_sync_judge_supported():
    def judge(_resp: str, _next_state: str) -> str:
        return "[HINT_START]return valid json next time[HINT_END]"

    ext = OPDHintExtractor(judge, OPDHintExtractorConfig(min_hint_chars=5))
    rec = _rec(next_state="bad json")
    ext.extract([rec])
    assert rec.metadata["opd_hint"] == "return valid json next time"


def test_max_records_budget_caps_judge_calls():
    calls = {"n": 0}

    def judge(_resp: str, _next_state: str) -> str:
        calls["n"] += 1
        return "a concrete actionable improvement hint"

    ext = OPDHintExtractor(judge, OPDHintExtractorConfig(max_records=1))
    recs = [_rec(next_state=f"signal {i}") for i in range(3)]
    stats = ext.extract(recs)
    assert calls["n"] == 1
    assert stats.n_judged == 1
    assert stats.n_hints_added == 1


# ---------------------------------------------------------------------------
# Factory behaviour
# ---------------------------------------------------------------------------


def test_factory_disabled_returns_none():
    assert build_opd_hint_extractor(None) is None
    assert build_opd_hint_extractor({}) is None
    assert build_opd_hint_extractor({"type": "disabled"}) is None


def test_factory_unknown_type_raises():
    import pytest

    with pytest.raises(ValueError):
        build_opd_hint_extractor({"type": "nonsense"})


# ---------------------------------------------------------------------------
# Full closed loop through the trainer
# ---------------------------------------------------------------------------


class _NextStateOnlyReward:
    """Reward that emits ONLY a next-state signal (no hint), mimicking a real
    env whose verifier reports feedback but does not pre-extract an OPD hint.
    The in-trainer extractor must recover the hint so OPD fires."""

    name = "next_state_only"

    async def evaluate(self, item: Any, trajectory: Any, tool_context: Any) -> Any:
        from hermes_agentic_rl.core.types import RewardResult

        runtime = trajectory.metadata.setdefault("runtime", {})
        runtime["next_state"] = "Wrong answer; the correct answer is 7"
        return RewardResult(name=self.name, score=0.5, weight=1.0, reason="ns")


def test_trainer_recovers_hint_and_fires_opd():
    from hermes_agentic_rl.algos.hybrid import HybridAlgo
    from hermes_agentic_rl.backends.tiny import (
        TinyBackendConfig,
        TinyCausalLMBackend,
    )
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.echo_task_env import (
        EchoTaskEnv,
        build_default_echo_dataset,
    )
    from hermes_agentic_rl.trainers.on_policy import (
        OnPolicyTrainer,
        OnPolicyTrainerConfig,
    )

    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([_NextStateOnlyReward()])  # type: ignore[list-item]
    cfg = OnPolicyTrainerConfig(
        n_iters=2,
        group_size=3,
        prompts_per_iter=1,
        max_new_tokens=4,
        log_every=100,
        seed=7,
        opd_teacher_fill=True,
        opd_hint_extractor={"type": "rule", "min_hint_chars": 5},
    )
    trainer = OnPolicyTrainer(policy=b, env=env, reward_manager=rm, algo=HybridAlgo(), cfg=cfg)
    stats = trainer.train()

    last = stats.iters[-1]
    # The in-trainer extractor recovered hints from the next-state signal...
    assert last.get("opd_hint_n_added", 0) > 0
    assert last.get("opd_hint_effective_rate", 0) > 0
    # ...the teacher filler then filled them...
    assert last.get("opd_teacher_n_filled", 0) > 0
    # ...and the OPD branch actually fired (no longer a no-op / GRPO-degraded).
    assert last.get("n_opd", 0) > 0
