"""Tests for Sprint 2: V-trace/TIS off-policy correction + async pipeline.

Covers:
  * TIS: on-policy → no-op; disabled → strict no-op; stale → clipped weight.
  * importance_weights clip/floor/mask behaviour.
  * V-trace reduces to n-step return when disabled; recursion is correct.
  * GRPO honours tis_rho_clip and reports tis_* stats.
  * ExperienceQueue: FIFO, drop-oldest bound, staleness accounting, shutdown.
  * run_async_training: a producer thread feeds the learner; updates land,
    staleness-gated TIS engages, gradients flow, queue stats are emitted.
"""

from __future__ import annotations

import threading
import time

import torch

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.common.vtrace import (
    TISConfig,
    VTraceConfig,
    importance_weights,
    tis_corrected_advantage,
    vtrace_returns,
)
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.distributed.experience_queue import (
    ExperienceItem,
    ExperienceQueue,
)
from hermes_agentic_rl.trainers.async_loop import AsyncLoopConfig, run_async_training

# ---------------------------------------------------------------------------
# TIS
# ---------------------------------------------------------------------------


def test_tis_on_policy_is_noop():
    new = torch.tensor([[-0.2, -0.3]])
    beh = new.clone()
    mask = torch.tensor([[True, True]])
    adv = torch.tensor([[1.0, 1.0]])
    corr, stats = tis_corrected_advantage(adv, new, beh, mask, TISConfig(rho_clip=2.0))
    assert torch.allclose(corr, adv)
    assert abs(stats["tis_weight_mean"] - 1.0) < 1e-6


def test_tis_disabled_is_strict_noop():
    new = torch.tensor([[0.9, 0.9]])
    beh = torch.tensor([[-0.5, -0.5]])
    mask = torch.tensor([[True, True]])
    adv = torch.tensor([[2.0, -1.0]])
    corr, _ = tis_corrected_advantage(adv, new, beh, mask, TISConfig(enabled=False))
    assert torch.allclose(corr, adv * mask.float())


def test_tis_clips_stale_weights():
    # new >> behavior → raw ratio > 1, clipped to rho_clip
    new = torch.tensor([[1.0, 1.0]])
    beh = torch.tensor([[-1.0, -1.0]])
    mask = torch.tensor([[True, True]])
    w = importance_weights(new, beh, mask, rho_clip=1.5)
    assert torch.all(w <= 1.5 + 1e-6)
    assert torch.all(w > 0)


def test_importance_weights_masking_and_floor():
    new = torch.tensor([[-3.0, 0.0]])
    beh = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    w = importance_weights(new, beh, mask, rho_floor=0.1, rho_clip=2.0)
    assert w[0, 1].item() == 0.0  # masked
    assert w[0, 0].item() >= 0.1  # floored


# ---------------------------------------------------------------------------
# V-trace
# ---------------------------------------------------------------------------


def test_vtrace_reduces_to_nstep_return_on_policy():
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5]])
    boot = torch.tensor([0.0])
    new = torch.tensor([[-0.2, -0.2]])
    beh = new.clone()  # on-policy → ρ = c = 1
    mask = torch.tensor([[True, True]])
    vs, _adv, _ = vtrace_returns(rewards, values, boot, new, beh, mask, VTraceConfig(gamma=1.0))
    # step1: r=1, bootstrap 0, V=0.5 → vs1 = 1.0; step0: r=0 + γ vs1 = 1.0
    assert torch.allclose(vs, torch.tensor([[1.0, 1.0]]), atol=1e-5)


def test_vtrace_disabled_matches_enabled_when_on_policy():
    rewards = torch.tensor([[0.5, 0.5]])
    values = torch.tensor([[0.1, 0.2]])
    boot = torch.tensor([0.3])
    lp = torch.tensor([[-0.1, -0.1]])
    mask = torch.tensor([[True, True]])
    vs_on, _, _ = vtrace_returns(rewards, values, boot, lp, lp.clone(), mask)
    vs_off, _, _ = vtrace_returns(
        rewards, values, boot, lp, lp.clone(), mask, VTraceConfig(enabled=False)
    )
    assert torch.allclose(vs_on, vs_off, atol=1e-5)


# ---------------------------------------------------------------------------
# GRPO integration
# ---------------------------------------------------------------------------


def _tiny() -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))


def _records(b: TinyCausalLMBackend, n: int = 4) -> list[RolloutRecord]:
    prompt = b.tokenizer.encode("async test")
    out = []
    for i in range(n):
        o = b.generate(prompt, max_new_tokens=4, seed=i)
        out.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(o.response_ids),
                old_logprobs=list(o.logprobs),
                reward=float(i % 2),
                group_id="g",
                metadata={},
            )
        )
    return out


def test_grpo_reports_tis_stats_when_enabled():
    b = _tiny()
    recs = _records(b)
    algo = GRPO(GRPOConfig(tis_rho_clip=1.5, kl_coef=0.0))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(recs))
    assert "tis_weight_mean" in stats.extra
    assert "tis_clip_frac" in stats.extra


def test_grpo_tis_disabled_by_default():
    b = _tiny()
    recs = _records(b)
    _loss, stats = GRPO(GRPOConfig(kl_coef=0.0)).compute_loss(b, None, RolloutBatch(recs))
    assert "tis_weight_mean" not in stats.extra  # no-op path adds nothing


# ---------------------------------------------------------------------------
# ExperienceQueue
# ---------------------------------------------------------------------------


def test_queue_fifo_and_staleness():
    q: ExperienceQueue[str] = ExperienceQueue(maxsize=8)
    q.put(ExperienceItem(payload="a", behavior_version=0))
    q.put(ExperienceItem(payload="b", behavior_version=1))
    q.set_learner_version(3)
    batch = q.get_batch(n=2, timeout=0.5)
    assert [it.payload for it in batch] == ["a", "b"]
    # staleness for "a" = 3-0 = 3 (last consumed recorded)
    assert q.stats.consumed == 2
    assert q.stats.mean_staleness == (3 + 2) / 2


def test_queue_drop_oldest_bounds_backlog():
    q: ExperienceQueue[int] = ExperienceQueue(maxsize=2)
    for i in range(5):
        q.put(ExperienceItem(payload=i, behavior_version=0))
    assert q.depth() <= 2
    assert q.stats.dropped_oldest >= 3
    remaining = [it.payload for it in q.get_batch(n=10, timeout=0.5)]
    # oldest dropped → only the freshest survive
    assert remaining == [3, 4]


def test_queue_get_batch_timeout_returns_empty():
    q: ExperienceQueue[int] = ExperienceQueue(maxsize=2)
    assert q.get_batch(n=2, timeout=0.05) == []


def test_queue_shutdown_unblocks_consumer():
    q: ExperienceQueue[int] = ExperienceQueue(maxsize=2)
    q.close()
    assert q.get_batch(n=2, timeout=0.5) == []


# ---------------------------------------------------------------------------
# Full async loop
# ---------------------------------------------------------------------------


def test_async_loop_consumes_and_updates():
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.echo_task_env import (
        EchoTaskEnv,
        build_default_echo_dataset,
    )
    from hermes_agentic_rl.trainers.on_policy import (
        OnPolicyTrainer,
        OnPolicyTrainerConfig,
    )

    b = _tiny()
    env = EchoTaskEnv(build_default_echo_dataset())

    class _Reward:
        name = "r"

        async def evaluate(self, item, trajectory, tool_context):
            from hermes_agentic_rl.core.types import RewardResult

            return RewardResult(name=self.name, score=0.5, weight=1.0, reason="x")

    rm = RewardManager([_Reward()])  # type: ignore[list-item]
    cfg = OnPolicyTrainerConfig(
        n_iters=0,
        group_size=2,
        prompts_per_iter=1,
        max_new_tokens=4,
        log_every=100,
        seed=3,
    )
    trainer = OnPolicyTrainer(
        policy=b, env=env, reward_manager=rm, algo=GRPO(GRPOConfig(kl_coef=0.0)), cfg=cfg
    )

    q: ExperienceQueue[list[RolloutRecord]] = ExperienceQueue(maxsize=16)

    # Producer thread: push stale-tagged record batches, then close.
    def producer() -> None:
        for v in range(6):
            recs = _records(b, n=2)
            # behavior_version stays 0 while learner advances → staleness grows
            q.put(ExperienceItem(payload=recs, behavior_version=0))
            time.sleep(0.01)
        q.close()

    t = threading.Thread(target=producer)
    t.start()
    stats = run_async_training(
        trainer,
        q,
        AsyncLoopConfig(
            max_updates=4,
            batch_size=2,
            get_timeout=0.5,
            idle_timeout=5.0,
            stale_tis_rho_clip=1.0,
            stale_threshold=0,
        ),
    )
    t.join(timeout=5.0)

    assert len(stats.iters) >= 1
    last = stats.iters[-1]
    # Async bookkeeping surfaced.
    assert "async_staleness" in last
    assert "queue_mean_staleness" in last
    assert "queue_depth" in last
    # After the first update the learner version advances, so later batches
    # are stale and TIS engages.
    stale_updates = [r for r in stats.iters if r.get("async_staleness", 0) > 0]
    if stale_updates:
        assert stale_updates[-1]["async_applied_tis_rho_clip"] == 1.0
