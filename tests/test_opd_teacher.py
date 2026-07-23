"""Tests for the OPD teacher-logprob closed loop (OpenClaw-RL §3.2).

Covers:
  * TeacherLogprobFiller fills aligned per-token teacher logprobs for hinted
    records and leaves hint-less records untouched.
  * Hint-axis classification + capability-axis-aware ``opd_adv_scale`` stamping.
  * OPDAlgo honours ``opd_adv_scale`` (the directive signal can be emphasised).
  * The filled records actually drive the OPD / Hybrid branch (no longer a
    no-op), i.e. gradients flow.
"""

from __future__ import annotations

from typing import Any

import torch

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.hybrid import HybridAlgo
from hermes_agentic_rl.algos.opd import OPDAlgo, OPDConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.rewards.opd_teacher import (
    TeacherFillConfig,
    TeacherLogprobFiller,
    classify_hint_axis,
)


def _make_tiny() -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))


def _make_records(
    b: TinyCausalLMBackend,
    n: int = 3,
    *,
    hint: str | None = "tool argument should be valid json",
) -> list[RolloutRecord]:
    prompt = b.tokenizer.encode("teacher fill test")
    records: list[RolloutRecord] = []
    for i in range(n):
        out = b.generate(prompt, max_new_tokens=4, seed=i)
        meta: dict[str, Any] = {}
        if hint is not None:
            meta["opd_hint"] = hint
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i),
                group_id="g0",
                metadata=meta,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Filler basics
# ---------------------------------------------------------------------------


def test_filler_fills_aligned_teacher_logprobs():
    b = _make_tiny()
    records = _make_records(b, n=3)
    filler = TeacherLogprobFiller(b, TeacherFillConfig())
    stats = filler.fill(records)

    assert stats.n_records == 3
    assert stats.n_with_hint == 3
    assert stats.n_filled == 3
    for rec in records:
        tl = rec.metadata.get("teacher_logprobs")
        assert isinstance(tl, list) and tl, "teacher_logprobs must be populated"
        # Token alignment: one teacher logprob per response token.
        assert len(tl) == len(rec.response_ids)


def test_filler_skips_records_without_hint():
    b = _make_tiny()
    records = _make_records(b, n=2, hint=None)
    filler = TeacherLogprobFiller(b, TeacherFillConfig())
    stats = filler.fill(records)

    assert stats.n_with_hint == 0
    assert stats.n_filled == 0
    for rec in records:
        assert "teacher_logprobs" not in rec.metadata


def test_filler_disabled_is_noop():
    b = _make_tiny()
    records = _make_records(b, n=2)
    filler = TeacherLogprobFiller(b, TeacherFillConfig(enabled=False))
    stats = filler.fill(records)
    assert stats.n_filled == 0
    for rec in records:
        assert "teacher_logprobs" not in rec.metadata


def test_filler_runs_under_no_grad():
    b = _make_tiny()
    records = _make_records(b, n=2)
    TeacherLogprobFiller(b, TeacherFillConfig()).fill(records)
    # No autograd graph should have been retained on policy params.
    assert all(p.grad is None for p in b.trainable_parameters())


# ---------------------------------------------------------------------------
# Hint-axis classification + capability-axis weighting
# ---------------------------------------------------------------------------


def test_classify_hint_axis_keyword_match():
    assert classify_hint_axis("the tool argument was wrong") == "tool_use_reliability"
    assert classify_hint_axis("you should finish the turn cleanly") == "interaction_control"
    assert classify_hint_axis("no relevant keywords here") is None


def test_filler_stamps_capability_axis_scale():
    b = _make_tiny()
    records = _make_records(b, n=2, hint="the tool argument must parse as json")
    filler = TeacherLogprobFiller(
        b,
        TeacherFillConfig(
            capability_axis_weights={"tool_use_reliability": 2.5},
            default_axis_weight=1.0,
        ),
    )
    stats = filler.fill(records)
    assert stats.axis_counts.get("tool_use_reliability") == 2
    for rec in records:
        assert rec.metadata.get("opd_adv_scale") == 2.5
        assert rec.metadata.get("opd_hint_axis") == "tool_use_reliability"


def test_filler_default_weight_for_unmatched_axis():
    b = _make_tiny()
    records = _make_records(b, n=1, hint="totally unrelated guidance text")
    filler = TeacherLogprobFiller(
        b,
        TeacherFillConfig(
            capability_axis_weights={"tool_use_reliability": 2.5},
            default_axis_weight=0.5,
        ),
    )
    filler.fill(records)
    assert records[0].metadata.get("opd_adv_scale") == 0.5
    assert "opd_hint_axis" not in records[0].metadata


# ---------------------------------------------------------------------------
# OPD honours the scale + closed loop drives gradients
# ---------------------------------------------------------------------------


def test_opd_advantage_scale_changes_loss():
    b = _make_tiny()
    base = _make_records(b, n=3)
    TeacherLogprobFiller(b, TeacherFillConfig()).fill(base)

    algo = OPDAlgo(OPDConfig(kl_coef=0.0))
    loss_unscaled, _ = algo.compute_loss(b, None, RolloutBatch(base))

    # Same records but with a non-trivial scale → loss magnitude must change.
    for rec in base:
        rec.metadata["opd_adv_scale"] = 3.0
    loss_scaled, stats_scaled = algo.compute_loss(b, None, RolloutBatch(base))

    assert stats_scaled.extra.get("opd_adv_scale_mean") == 3.0
    assert not torch.allclose(loss_unscaled, loss_scaled)


def test_filled_records_drive_opd_gradient():
    b = _make_tiny()
    records = _make_records(b, n=3)
    # Before filling, OPD finds no hints → no OPD term.
    TeacherLogprobFiller(b, TeacherFillConfig()).fill(records)
    algo = OPDAlgo(OPDConfig(kl_coef=0.0))
    loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_with_hints"] == 3
    if loss.requires_grad:
        loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )


def test_hybrid_uses_filled_teacher_logprobs():
    b = _make_tiny()
    records = _make_records(b, n=4)
    # Without the filler the hybrid OPD branch would be empty.
    TeacherLogprobFiller(b, TeacherFillConfig()).fill(records)
    algo = HybridAlgo()
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_opd"] == 4


# ---------------------------------------------------------------------------
# Full closed loop through the trainer (config -> filler -> OPD branch fires)
# ---------------------------------------------------------------------------


class _HintInjectingReward:
    """Reward component that scores echoes AND writes an OPD hint, mimicking
    NextStatePRMComponent's directive-signal path."""

    name = "hint_injector"

    async def evaluate(self, item: Any, trajectory: Any, tool_context: Any) -> Any:
        from hermes_agentic_rl.core.types import RewardResult

        runtime = trajectory.metadata.setdefault("runtime", {})
        rl = runtime.setdefault("rl", {})
        rl["opd_hint"] = "the tool argument should be valid json"
        return RewardResult(name=self.name, score=0.5, weight=1.0, reason="hint")


def test_trainer_closed_loop_fills_and_fires_opd():
    from hermes_agentic_rl.algos.hybrid import HybridAlgo
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.echo_task_env import (
        EchoTaskEnv,
        build_default_echo_dataset,
    )
    from hermes_agentic_rl.trainers.on_policy import (
        OnPolicyTrainer,
        OnPolicyTrainerConfig,
    )

    b = _make_tiny()
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([_HintInjectingReward()])  # type: ignore[list-item]
    cfg = OnPolicyTrainerConfig(
        n_iters=2,
        group_size=3,
        prompts_per_iter=1,
        max_new_tokens=4,
        log_every=100,
        seed=7,
        opd_teacher_fill=True,
        opd_capability_axis_weights={"tool_use_reliability": 2.0},
    )
    trainer = OnPolicyTrainer(
        policy=b,
        env=env,
        reward_manager=rm,
        algo=HybridAlgo(),
        cfg=cfg,
    )
    stats = trainer.train()

    assert len(stats.iters) == 2
    last = stats.iters[-1]
    # The teacher filler ran and filled every hinted record.
    assert last.get("opd_teacher_n_filled", 0) > 0
    assert last.get("opd_teacher_n_with_hint", 0) == last.get("opd_teacher_n_filled", 0)
    # The OPD branch actually fired (no longer a no-op).
    assert last.get("n_opd", 0) > 0
    # Capability-axis weighting was applied for the tool-use hint.
    assert last.get("opd_teacher_axis/tool_use_reliability", 0) > 0
