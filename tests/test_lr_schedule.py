"""Tests for LR scheduler integration and grad-accum awareness."""

from __future__ import annotations

import pytest

from hermes_agentic_rl.trainers.lr_schedule import (
    ConstantLR,
    CosineLR,
    LinearLR,
    make_lr_scheduler,
)


class TestConstantLR:
    def test_always_returns_base_lr(self):
        sched = ConstantLR(lr=1e-3)
        assert sched.get_lr(0) == 1e-3
        assert sched.get_lr(100) == 1e-3


class TestLinearLR:
    def test_warmup_phase(self):
        sched = LinearLR(lr=1e-3, warmup_steps=10, total_steps=100)
        assert sched.get_lr(0) == 0.0
        assert abs(sched.get_lr(5) - 5e-4) < 1e-8
        assert sched.get_lr(10) == 1e-3

    def test_decay_phase(self):
        sched = LinearLR(lr=1e-3, warmup_steps=0, total_steps=100, end_lr=0.0)
        assert sched.get_lr(0) == 1e-3
        assert sched.get_lr(50) == 5e-4
        assert sched.get_lr(100) == 0.0

    def test_flat_tail(self):
        sched = LinearLR(lr=1e-3, warmup_steps=0, total_steps=100, end_lr=1e-5)
        assert sched.get_lr(200) == 1e-5


class TestCosineLR:
    def test_cosine_decay(self):
        sched = CosineLR(lr=1e-3, total_steps=100, warmup_steps=0, end_lr=0.0)
        assert sched.get_lr(0) == 1e-3
        assert sched.get_lr(50) < 1e-3
        assert sched.get_lr(50) > 0.0
        assert abs(sched.get_lr(100)) < 1e-8

    def test_warmup_then_cosine(self):
        sched = CosineLR(lr=1e-3, total_steps=100, warmup_steps=10, end_lr=0.0)
        assert sched.get_lr(0) == 0.0
        assert sched.get_lr(10) == 1e-3
        assert sched.get_lr(55) < 1e-3


class TestMakeLRScheduler:
    def test_constant(self):
        sched = make_lr_scheduler("constant", lr=5e-4)
        assert isinstance(sched, ConstantLR)
        assert sched.get_lr(0) == 5e-4

    def test_linear(self):
        sched = make_lr_scheduler("linear", lr=1e-3, warmup_steps=5, total_steps=50)
        assert isinstance(sched, LinearLR)

    def test_cosine(self):
        sched = make_lr_scheduler("cosine", lr=1e-3, total_steps=100)
        assert isinstance(sched, CosineLR)

    def test_warmup_cosine_alias(self):
        sched = make_lr_scheduler("warmup_cosine", lr=1e-3, total_steps=100, warmup_steps=10)
        assert isinstance(sched, CosineLR)
        assert sched.warmup_steps == 10

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LR scheduler"):
            make_lr_scheduler("nonexistent")


class TestGradAccumAwareStepping:
    """Verify that LR scheduling counts optimizer steps, not iterations.

    With grad_accum_steps=2, each iteration produces 2 optimizer steps.
    The LR should advance by 2 steps per iteration.
    """

    def test_lr_tracks_actual_steps(self):
        sched = make_lr_scheduler("linear", lr=1e-3, warmup_steps=10, total_steps=100)
        # Simulating grad_accum=2: step counter advances by 2 per iter
        step_counter = 0
        lrs = []
        for _iter in range(5):
            for _micro in range(2):  # 2 optimizer steps per iter
                step_counter += 1
                lrs.append(sched.get_lr(step_counter))
        # After 10 steps, warmup is complete → LR reaches peak
        assert lrs[-1] == 1e-3  # step 10 = end of warmup
