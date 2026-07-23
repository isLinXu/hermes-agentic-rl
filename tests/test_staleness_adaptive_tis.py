"""Staleness-adaptive TIS controller tests: schedule, adaptation, stats."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.common.staleness_adaptive_tis import (
    StalenessAdaptiveTIS,
    StalenessSchedule,
)
from hermes_agentic_rl.algos.common.vtrace import (
    TISConfig,
    tis_corrected_advantage,
)

# ── StalenessSchedule tests ──


def test_schedule_zero_staleness_gives_max_clip():
    sched = StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10)
    assert sched.clip_at(0) == 2.0
    assert sched.clip_at(-1) == 2.0  # negative treated as 0


def test_schedule_max_staleness_gives_min_clip():
    sched = StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10)
    assert sched.clip_at(10) == 1.0
    assert sched.clip_at(15) == 1.0  # clamped


def test_schedule_linear_interpolation_midpoint():
    sched = StalenessSchedule(
        max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10, interpolation="linear",
    )
    # At staleness=5, should be midpoint: 1.5
    assert sched.clip_at(5) == pytest.approx(1.5)


def test_schedule_exp_interpolation_decreases():
    sched = StalenessSchedule(
        max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10, interpolation="exp",
    )
    clip_0 = sched.clip_at(0)
    clip_5 = sched.clip_at(5)
    clip_10 = sched.clip_at(10)
    assert clip_0 > clip_5 > clip_10
    assert clip_0 == pytest.approx(2.0)
    assert clip_10 == pytest.approx(1.0)


# ── StalenessAdaptiveTIS tests ──


def test_controller_initial_clip_is_max():
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0),
    )
    assert controller.current_clip == 2.0
    cfg = controller.get_config()
    assert cfg.rho_clip == 2.0
    assert cfg.enabled is True


def test_controller_observe_updates_clip():
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
    )
    controller.observe_staleness(5)
    assert controller.current_clip == pytest.approx(1.5)
    controller.observe_staleness(10)
    # Mean of [5, 10] = 7.5 → clip should be between 1.0 and 1.5
    assert 1.0 < controller.current_clip < 1.5


def test_controller_rolling_average():
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
        window_size=3,
    )
    controller.observe_staleness(10)  # mean=10 → clip=1.0
    controller.observe_staleness(0)   # mean=5 → clip=1.5
    controller.observe_staleness(0)   # mean=3.33 → clip≈1.667
    assert controller.mean_staleness == pytest.approx(10 / 3, abs=0.01)
    assert controller.current_clip > 1.5  # should be closer to max


def test_controller_disabled_returns_identity_config():
    controller = StalenessAdaptiveTIS(enabled=False)
    cfg = controller.get_config()
    assert cfg.enabled is False


def test_controller_last_staleness():
    controller = StalenessAdaptiveTIS()
    controller.observe_staleness(3)
    controller.observe_staleness(7)
    assert controller.last_staleness == 7.0


def test_controller_stats():
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
    )
    controller.observe_staleness(5)
    stats = controller.stats()
    assert "tis_adaptive_clip" in stats
    assert "tis_adaptive_mean_staleness" in stats
    assert stats["tis_adaptive_mean_staleness"] == pytest.approx(5.0)
    assert stats["tis_adaptive_steps"] == 1.0


def test_controller_reset():
    controller = StalenessAdaptiveTIS()
    controller.observe_staleness(5)
    controller.observe_staleness(7)
    assert controller.step_count == 2
    controller.reset()
    assert controller.step_count == 0
    assert controller.mean_staleness == 0.0
    assert controller.current_clip == controller.schedule.max_rho_clip


def test_controller_window_evicts_old():
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
        window_size=2,
    )
    controller.observe_staleness(0)   # history=[0]
    controller.observe_staleness(0)   # history=[0, 0], mean=0, clip=2.0
    assert controller.current_clip == pytest.approx(2.0)
    controller.observe_staleness(10)  # history=[0, 10] (oldest 0 evicted? no, [0,10])
    # window=2: [0, 10] → mean=5 → clip=1.5
    assert controller.mean_staleness == pytest.approx(5.0)
    controller.observe_staleness(10)  # history=[10, 10], mean=10, clip=1.0
    assert controller.current_clip == pytest.approx(1.0)


# ── Integration: adaptive TIS with tis_corrected_advantage ──


def test_adaptive_tis_integration_with_advantage():
    """End-to-end: adaptive controller produces config used by tis_corrected_advantage."""
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
    )
    # Simulate high staleness → conservative clip.
    controller.observe_staleness(10)
    cfg = controller.get_config()
    assert cfg.rho_clip == pytest.approx(1.0)

    # Create tensors.
    B, T = 2, 4
    advantage = torch.randn(B, T)
    new_logprobs = torch.randn(B, T)
    behavior_logprobs = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)

    corrected, stats = tis_corrected_advantage(
        advantage, new_logprobs, behavior_logprobs, mask, cfg,
    )
    assert corrected.shape == (B, T)
    assert "tis_weight_mean" in stats
    assert "tis_clip_frac" in stats


def test_adaptive_tis_zero_staleness_max_clip():
    """When staleness=0, clip should be max → more signal from on-policy data."""
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
    )
    controller.observe_staleness(0)
    cfg = controller.get_config()
    assert cfg.rho_clip == pytest.approx(2.0)
    assert cfg.enabled is True


def test_adaptive_tis_increasing_staleness_decreases_clip():
    """As staleness increases over steps, clip should decrease monotonically."""
    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(max_rho_clip=2.0, min_rho_clip=1.0, max_staleness=10),
        window_size=1,  # react immediately
    )
    clips = []
    for s in range(0, 11):
        controller.observe_staleness(s)
        clips.append(controller.current_clip)

    # Should be monotonically non-increasing.
    for i in range(1, len(clips)):
        assert clips[i] <= clips[i - 1] + 1e-8, f"Clip increased at step {i}: {clips}"
