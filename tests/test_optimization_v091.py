"""Tests for the v0.9.1 / v0.9.2 framework optimizations.

Covers:

* RolloutManager → ``assistant_message`` extraction (positional + explicit).
* TrainerBridge → retry, backoff counters, cancellation propagation.
* ToolcallReward → custom weight breakdown.
* ``build_shared_on_policy_config`` field forwarding.
* ``validate_config`` accepts / rejects schemas.
* RolloutBatch shared-logprobs cache + ``stack_cached_logprobs``.
* ``old_logprobs_tensor`` zero-copy path.

These tests run without GPU and without torch model construction; they
exercise the abstractions added in this revision in isolation.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from hermes_agentic_rl.config import ConfigValidationError, validate_config
from hermes_agentic_rl.core.rollout_manager import (
    _extract_assistant_messages_by_turn,
)
from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.core.types import RewardSummary, RolloutStep, Trajectory
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig
from hermes_agentic_rl.trainers.on_policy_config import (
    OnPolicyTrainerConfig,
    build_shared_on_policy_config,
)
from hermes_agentic_rl.trainers.ppo_trainer import PPOTrainerConfig


# ---------------------------------------------------------------------------
# RolloutManager assistant message extraction
# ---------------------------------------------------------------------------


def test_assistant_messages_positional_fallback():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "second"},
    ]
    out = _extract_assistant_messages_by_turn(messages, turn_count=2)
    assert out == ["first", "second"]


def test_assistant_messages_explicit_turn_index_wins():
    messages = [
        {"role": "assistant", "content": "noise"},
        {"role": "assistant", "content": "turn0", "turn_index": 0},
        {"role": "assistant", "content": "turn1", "turn_index": 1},
    ]
    out = _extract_assistant_messages_by_turn(messages, turn_count=2)
    assert out == ["turn0", "turn1"]


def test_assistant_messages_truncates_extra_turns():
    out = _extract_assistant_messages_by_turn([], turn_count=3)
    assert out == [None, None, None]


def test_assistant_messages_drops_non_string_content():
    messages = [{"role": "assistant", "content": 42, "turn_index": 0}]
    out = _extract_assistant_messages_by_turn(messages, turn_count=1)
    assert out == [None]


# ---------------------------------------------------------------------------
# TrainerBridge retry + metrics
# ---------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self, *, fail_first_n: int = 0) -> None:
        self.fail_first_n = fail_first_n
        self.calls = 0

    async def submit(self, item, trajectory, reward_summary):
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise RuntimeError(f"transient failure {self.calls}")
        return {"ok": True, "calls": self.calls}


def _dummy_traj_and_summary():
    traj = Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output=None,
        finished_naturally=True,
        turns_used=0,
    )
    summary = RewardSummary(final_score=0.0, components=[])
    return traj, summary


def test_trainer_bridge_happy_path():
    bridge = TrainerBridge(_FakeTrainer(), max_retries=1)
    traj, summary = _dummy_traj_and_summary()
    out = asyncio.run(bridge.submit({}, traj, summary))
    assert out == {"ok": True, "calls": 1}
    assert bridge.metrics()["submitted"] == 1
    assert bridge.metrics()["failed"] == 0
    assert bridge.metrics()["retried"] == 0


def test_trainer_bridge_retry_then_succeed():
    bridge = TrainerBridge(
        _FakeTrainer(fail_first_n=2),
        max_retries=3,
        backoff_base=0.0,
        backoff_max=0.0,
    )
    traj, summary = _dummy_traj_and_summary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = asyncio.run(bridge.submit({}, traj, summary))
    assert out["calls"] == 3
    m = bridge.metrics()
    assert m["submitted"] == 1 and m["retried"] == 2 and m["failed"] == 0


def test_trainer_bridge_exhausts_retries():
    bridge = TrainerBridge(
        _FakeTrainer(fail_first_n=99),
        max_retries=2,
        backoff_base=0.0,
    )
    traj, summary = _dummy_traj_and_summary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="transient failure"):
            asyncio.run(bridge.submit({}, traj, summary))
    m = bridge.metrics()
    assert m["failed"] == 1 and m["retried"] == 1
    assert m["last_error"] is not None and "transient" in m["last_error"]


def test_trainer_bridge_propagates_cancellation():
    class CancelTrainer:
        async def submit(self, *_a, **_k):
            raise asyncio.CancelledError()

    bridge = TrainerBridge(CancelTrainer(), max_retries=5)
    traj, summary = _dummy_traj_and_summary()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(bridge.submit({}, traj, summary))
    assert bridge.metrics()["retried"] == 0


def test_trainer_bridge_rejects_zero_retries():
    with pytest.raises(ValueError):
        TrainerBridge(_FakeTrainer(), max_retries=0)


# ---------------------------------------------------------------------------
# ToolcallReward weight knobs
# ---------------------------------------------------------------------------


def test_toolcall_reward_default_weights_preserve_legacy_blend():
    r = ToolcallReward()
    assert abs(r.name_weight - 0.4) < 1e-9
    assert abs(r.schema_weight - 0.3) < 1e-9
    assert abs(r.value_weight - 0.3) < 1e-9


def test_toolcall_reward_custom_weights_normalize():
    r = ToolcallReward(name_weight=2.0, schema_weight=1.0, value_weight=1.0)
    assert abs(r.name_weight - 0.5) < 1e-9
    assert abs(r.schema_weight - 0.25) < 1e-9
    assert abs(r.value_weight - 0.25) < 1e-9


def test_toolcall_reward_zero_weights_fall_back_to_default():
    r = ToolcallReward(name_weight=0.0, schema_weight=0.0, value_weight=0.0)
    assert abs(r.name_weight - 0.4) < 1e-9
    assert abs(r.schema_weight - 0.3) < 1e-9
    assert abs(r.value_weight - 0.3) < 1e-9


def test_toolcall_reward_empty_calls_still_zero():
    traj = Trajectory(
        task_id="t",
        prompt="p",
        steps=[RolloutStep(turn_index=0, tool_calls=[])],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )
    result = asyncio.run(
        ToolcallReward(weight=0.3).evaluate({}, traj, tool_context=None)
    )
    assert result.score == 0.0
    assert "no tool calls" in result.reason


# ---------------------------------------------------------------------------
# build_shared_on_policy_config
# ---------------------------------------------------------------------------


def test_build_shared_copies_overlapping_fields_grpo():
    cfg = GRPOTrainerConfig(n_iters=7, group_size=3, lr=2e-4, max_new_tokens=12)
    shared = build_shared_on_policy_config(cfg)
    assert isinstance(shared, OnPolicyTrainerConfig)
    assert shared.n_iters == 7
    assert shared.group_size == 3
    assert abs(shared.lr - 2e-4) < 1e-12
    assert shared.max_new_tokens == 12
    # Algo-specific field must NOT leak into the shared dataclass.
    assert not hasattr(shared, "clip_eps")


def test_build_shared_copies_overlapping_fields_ppo():
    cfg = PPOTrainerConfig(n_iters=11, lr=3e-4, gamma=0.9)
    shared = build_shared_on_policy_config(cfg)
    assert shared.n_iters == 11
    assert abs(shared.lr - 3e-4) < 1e-12
    # PPO-only field gamma should not appear on shared
    assert not hasattr(shared, "gamma")


def test_build_shared_works_with_plain_object():
    class _Bag:
        pass

    bag = _Bag()
    bag.n_iters = 5
    bag.lr = 1e-3
    shared = build_shared_on_policy_config(bag)
    assert shared.n_iters == 5
    # Fields not provided fall back to OnPolicyTrainerConfig defaults.
    assert shared.group_size == OnPolicyTrainerConfig().group_size


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def test_validate_config_accepts_clean_input():
    errs = validate_config(
        {
            "runtime": {"integration": "fake", "max_agent_turns": 5},
            "trainer": {"n_iters": 10, "lr": 1e-3},
            "reward": {"aggregator": "weighted_sum"},
        },
        strict=False,
    )
    assert errs == []


def test_validate_config_rejects_unknown_integration():
    with pytest.raises(ConfigValidationError):
        validate_config({"runtime": {"integration": "ghost", "max_agent_turns": 1}})


def test_validate_config_rejects_zero_max_turns():
    with pytest.raises(ConfigValidationError):
        validate_config({"runtime": {"integration": "fake", "max_agent_turns": 0}})


def test_validate_config_rejects_bad_trainer():
    with pytest.raises(ConfigValidationError):
        validate_config(
            {
                "runtime": {"integration": "fake", "max_agent_turns": 1},
                "trainer": {"n_iters": 0, "lr": -1.0},
            }
        )


# ---------------------------------------------------------------------------
# RolloutBatch shared-logprobs cache (no torch model required)
# ---------------------------------------------------------------------------


def test_stack_cached_logprobs_preserves_graph_and_shapes():
    pytest.importorskip("torch")
    import torch

    from hermes_agentic_rl.algos.base import (
        RolloutBatch,
        RolloutRecord,
        stack_cached_logprobs,
    )

    rec_a = RolloutRecord(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        old_logprobs=[-0.1, -0.2, -0.3],
        reward=0.5,
        group_id="g",
    )
    rec_b = RolloutRecord(
        prompt_ids=[1],
        response_ids=[7],
        old_logprobs=[-0.4],
        reward=0.1,
        group_id="g",
    )

    # Pretend these came from a single batched forward (B=2, T=3):
    base = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.0, 0.0]], requires_grad=True)
    cache = {id(rec_a): base[0, :3], id(rec_b): base[1, :1]}
    batch = RolloutBatch(
        records=[rec_a, rec_b],
        shared_new_logprobs=cache,
        shared_logprobs_temperature=1.0,
    )
    assert batch.has_shared_logprobs_for([rec_a, rec_b])
    assert not batch.has_shared_logprobs_for(
        [rec_a, RolloutRecord(prompt_ids=[], response_ids=[], old_logprobs=[],
                              reward=0.0, group_id="g")]
    )

    logp, mask = stack_cached_logprobs(cache, [rec_a, rec_b])
    assert logp.shape == (2, 3)
    assert mask.shape == (2, 3)
    assert bool(mask[0, 0]) and bool(mask[0, 1]) and bool(mask[0, 2])
    assert bool(mask[1, 0]) and not bool(mask[1, 1]) and not bool(mask[1, 2])

    # Gradient must flow back to the original tensor.
    logp.sum().backward()
    assert base.grad is not None
    # rec_a contributes [1,1,1], rec_b contributes [1,0,0] → matches mask.
    assert torch.allclose(
        base.grad,
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
    )


def test_old_logprobs_tensor_uses_cached_metadata():
    pytest.importorskip("torch")
    import torch

    from hermes_agentic_rl.algos.base import RolloutRecord, old_logprobs_tensor

    cached = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)
    rec = RolloutRecord(
        prompt_ids=[1],
        response_ids=[2, 3, 4],
        old_logprobs=[-0.1, -0.2, -0.3],
        reward=0.0,
        group_id="g",
        metadata={"_old_logprobs_tensor": cached},
    )
    out = old_logprobs_tensor(rec, length=3, dtype=torch.float32, device=torch.device("cpu"))
    assert torch.allclose(out, cached)

    # Falls back to list when cache is absent.
    rec2 = RolloutRecord(
        prompt_ids=[1],
        response_ids=[2, 3, 4],
        old_logprobs=[-0.5, -0.6, -0.7],
        reward=0.0,
        group_id="g",
    )
    out2 = old_logprobs_tensor(
        rec2, length=3, dtype=torch.float32, device=torch.device("cpu"),
    )
    assert torch.allclose(
        out2, torch.tensor([-0.5, -0.6, -0.7], dtype=torch.float32)
    )

    # Length shorter than cached → right-aligned slice.
    out3 = old_logprobs_tensor(
        rec, length=2, dtype=torch.float32, device=torch.device("cpu"),
    )
    assert torch.allclose(out3, torch.tensor([-0.2, -0.3], dtype=torch.float32))
