"""Multi-process rollout pool — end-to-end smoke.

We run a tiny 2-worker pool over the echo env for a handful of rollouts and
assert records come back with the right shape.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.distributed import MPRolloutPool, MPRolloutPoolConfig, RolloutTask


def _build_worker(ctx: dict):
    """Module-level so `spawn` can pickle it."""
    from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.echo_task_env import (
        EchoRewardComponent,
        EchoTaskEnv,
        build_default_echo_dataset,
    )

    backend = TinyCausalLMBackend(
        TinyBackendConfig(seed=ctx.get("seed", 0), dim=16, n_heads=2, n_layers=1, max_len=32)
    )
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])

    def _factory(*, backend, seed):
        return PolicyAgentLoop(
            backend=backend,
            max_new_tokens=4,
            temperature=1.0,
            seed=seed,
        )

    return backend, env, rm, _factory


def test_mp_rollout_pool_runs_and_returns_records():
    pool = MPRolloutPool(
        MPRolloutPoolConfig(n_workers=2, ctx_method="spawn", task_timeout=60.0),
        builder_fn=_build_worker,
    )
    pool.start()
    try:
        # broadcast an empty state dict — workers still run with their local
        # backend weights. The purpose here is to validate the wire/queue path.
        import torch as _t

        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        dummy = TinyCausalLMBackend(
            TinyBackendConfig(seed=0, dim=16, n_heads=2, n_layers=1, max_len=32)
        )
        pool.broadcast_weights({k: v.detach().cpu() for k, v in dummy.model.state_dict().items()})
        tasks = [
            RolloutTask(
                task_id=f"echo-{i}",
                item={"task_id": f"echo-{i}", "instruction": "Say: hi", "target": "hi"},
                instruction="Say: hi",
                seed=42 + i,
                task_seq=i,
            )
            for i in range(4)
        ]
        pool.submit_tasks(tasks)
        results = pool.drain(expected=len(tasks))
        assert len(results) == 4
        assert [r["task_seq"] for r in results] == [0, 1, 2, 3]
        for r in results:
            assert "records" in r and len(r["records"]) >= 1
            first = r["records"][0]
            assert "prompt_ids" in first and "response_ids" in first and "reward" in first
    finally:
        pool.shutdown()
