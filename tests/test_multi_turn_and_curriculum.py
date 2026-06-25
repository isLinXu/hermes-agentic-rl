"""Integration-ish tests for MultiTurnAgentLoop, SimToolEnv, CurriculumEnv."""

from __future__ import annotations

import asyncio
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.agent_loop.multi_turn_loop import MultiTurnAgentLoop
from hermes_agentic_rl.algos.common.advantage import group_normalize_advantage
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.types import RewardResult
from hermes_agentic_rl.envs.curriculum import CurriculumEnv
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.envs.sim_tool_env import (
    DEFAULT_TOOLS,
    SimToolEnv,
    SimToolRewardComponent,
    build_sim_tool_dataset,
    safe_eval,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from hermes_agentic_rl.trainers.multi_turn_credit import (
    assign_multi_turn_rewards,
    parse_multi_turn_credit_config,
)


def _tiny(seed: int = 0, with_value: bool = False) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(
            seed=seed, with_value_head=with_value, dim=16, n_heads=2, n_layers=1, max_len=64
        )
    )


def test_safe_eval_small_subset_only():
    assert safe_eval("3 + 4") == "7"
    assert safe_eval("2 * (3 + -4)") == "-2"
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(ValueError):
        safe_eval("pow(2, 3)")


def test_multi_turn_loop_emits_per_turn_metadata():
    backend = _tiny()
    loop = MultiTurnAgentLoop(
        backend=backend,
        tools=DEFAULT_TOOLS,
        max_turns=2,
        max_new_tokens_per_turn=6,
        temperature=1.0,
        seed=1,
    )
    out = asyncio.run(loop.run("What is 1 + 2?"))
    rl = out["metadata"]["rl"]
    assert rl["multi_turn"] is True
    assert "turns" in rl
    assert len(rl["turns"]) >= 1
    # flat views are consistent with per-turn concatenation
    flat = [tok for t in rl["turns"] for tok in t["response_ids"]]
    assert flat == list(rl["response_ids"])


def test_multi_turn_loop_attaches_tool_calls_to_assistant_messages():
    class FakeTokenizer:
        bos_id = 1
        eos_id = 0
        pad_id = 0

        def encode(self, text: str, add_eos: bool = False) -> list[int]:
            out = [ord(ch) for ch in text]
            if add_eos:
                out.append(0)
            return out

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

    class FakeBackend:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()
            self.calls = 0

        def generate(self, prompt_ids, max_new_tokens, temperature=1.0, seed=None, **kwargs):
            del prompt_ids, max_new_tokens, temperature, seed, kwargs
            self.calls += 1
            text = "<tool_call>calc(1 + 2)</tool_call>" if self.calls == 1 else "answer=3"
            response_ids = self.tokenizer.encode(text)
            return type(
                "Gen",
                (),
                {
                    "response_ids": response_ids,
                    "logprobs": [-0.1] * len(response_ids),
                    "finished": self.calls > 1,
                },
            )()

    loop = MultiTurnAgentLoop(
        backend=FakeBackend(),  # type: ignore[arg-type]
        tools=DEFAULT_TOOLS,
        max_turns=2,
        max_new_tokens_per_turn=32,
        temperature=1.0,
        seed=0,
    )
    out = asyncio.run(loop.run("What is 1 + 2?"))
    assistant_messages = [m for m in out["messages"] if m.get("role") == "assistant"]

    assert len(assistant_messages) == 2
    assert assistant_messages[0]["tool_calls"][0]["name"] == "calc"
    assert assistant_messages[1].get("tool_calls") is None


def test_multi_turn_credit_hybrid_uses_local_tool_feedback():
    from hermes_agentic_rl.core.types import RolloutStep, Trajectory

    trajectory = Trajectory(
        task_id="calc-1",
        prompt="What is 1 + 2?",
        steps=[
            RolloutStep(
                turn_index=0,
                tool_calls=[{"name": "calc", "arg": "1 + 2"}],
                tool_results=[{"name": "calc", "result": "3"}],
            ),
            RolloutStep(turn_index=1),
        ],
        final_output="answer=3",
        finished_naturally=True,
        turns_used=2,
        metadata={
            "messages": [
                {"role": "user", "content": "What is 1 + 2?"},
                {
                    "role": "assistant",
                    "content": "<tool_call>calc(1 + 2)</tool_call>",
                    "tool_calls": [{"name": "calc", "arg": "1 + 2"}],
                },
                {"role": "tool", "name": "calc", "content": "success: 3"},
                {"role": "assistant", "content": "answer=3"},
            ]
        },
    )

    turn_rewards = assign_multi_turn_rewards(
        trajectory,
        final_reward=1.0,
        n_turns=2,
        cfg={
            "mode": "hybrid",
            "gamma": 0.9,
            "final_reward_weight": 0.4,
            "local_reward_weight": 0.6,
        },
    )

    assert len(turn_rewards) == 2
    assert turn_rewards[0]["local_component"] > 0.0
    assert turn_rewards[1]["local_component"] == 0.0
    assert turn_rewards[0]["reward"] > turn_rewards[1]["reward"]


def test_multi_turn_credit_uses_teacher_similarity_for_final_turn():
    from hermes_agentic_rl.core.types import RolloutStep, Trajectory

    trajectory = Trajectory(
        task_id="calc-1",
        prompt="What is 1 + 2?",
        steps=[
            RolloutStep(
                turn_index=0,
                tool_calls=[{"name": "calc", "arg": "1 + 2"}],
                tool_results=[{"name": "calc", "result": "3"}],
            ),
            RolloutStep(turn_index=1),
        ],
        final_output="answr=3",
        finished_naturally=True,
        turns_used=2,
        metadata={
            "messages": [
                {"role": "user", "content": "What is 1 + 2?"},
                {
                    "role": "assistant",
                    "content": "<tool_call>calc(1 + 2)</tool_call>",
                    "tool_calls": [{"name": "calc", "arg": "1 + 2"}],
                },
                {"role": "tool", "name": "calc", "content": "3"},
                {"role": "assistant", "content": "answr=3"},
            ]
        },
    )

    turn_rewards = assign_multi_turn_rewards(
        trajectory,
        final_reward=0.7,
        n_turns=2,
        cfg={"mode": "hybrid", "gamma": 0.9},
        teacher_responses=["<tool_call>calc(1 + 2)</tool_call>", "answer=3"],
    )

    assert turn_rewards[1]["local_component"] > 0.0
    teacher_meta = turn_rewards[1]["judge_metadata"]["teacher_metadata"]
    assert teacher_meta["expected_response"] == "answer=3"
    assert teacher_meta["similarity"] > 0.0


def test_grpo_trainer_multi_turn_credit_flows_into_batch():
    class ConstantReward:
        name = "constant"

        async def evaluate(self, item, trajectory, tool_context):
            del item, trajectory, tool_context
            return RewardResult(name=self.name, score=1.0, reason="constant", weight=1.0)

    class FakeLoop:
        async def run(self, prompt: str):
            del prompt
            return {
                "messages": [
                    {"role": "user", "content": "What is 1 + 2?"},
                    {
                        "role": "assistant",
                        "content": "<tool_call>calc(1 + 2)</tool_call>",
                        "tool_calls": [{"name": "calc", "arg": "1 + 2"}],
                    },
                    {"role": "tool", "name": "calc", "content": "success: 3"},
                    {"role": "assistant", "content": "answer=3"},
                ],
                "tool_calls": [[{"name": "calc", "arg": "1 + 2"}], []],
                "tool_results": [[{"name": "calc", "result": "success: 3"}], []],
                "final_output": "answer=3",
                "finished_naturally": True,
                "turns_used": 2,
                "metadata": {
                    "runtime": "multi_turn_agent_loop",
                    "prompt": "What is 1 + 2?",
                    "rl": {
                        "prompt_ids": [1, 2, 3],
                        "response_ids": [4, 5, 6, 7],
                        "old_logprobs": [-0.1, -0.1, -0.1, -0.1],
                        "multi_turn": True,
                        "turns": [
                            {
                                "prompt_prefix_ids": [1, 2, 3],
                                "response_ids": [4, 5],
                                "old_logprobs": [-0.1, -0.1],
                            },
                            {
                                "prompt_prefix_ids": [1, 2, 3, 4, 5, 8],
                                "response_ids": [6, 7],
                                "old_logprobs": [-0.1, -0.1],
                            },
                        ],
                    },
                },
            }

    env = EchoTaskEnv(
        [{"task_id": "calc-1", "instruction": "What is 1 + 2?", "target": "answer=3"}]
    )
    rm = RewardManager([ConstantReward()])  # type: ignore[list-item]
    trainer = GRPOTrainer(
        policy=_tiny(),
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=2,
            prompts_per_iter=1,
            lr=1e-3,
            max_new_tokens=4,
            temperature=1.0,
            log_every=100,
            seed=0,
            multi_turn=True,
            multi_turn_credit={
                "mode": "hybrid",
                "gamma": 0.9,
                "final_reward_weight": 0.4,
                "local_reward_weight": 0.6,
            },
        ),
        agent_loop_factory=lambda backend, seed: FakeLoop(),  # type: ignore[arg-type]
    )

    seen_batches: list[list[tuple[int, float, str, str, str]]] = []
    original = trainer.algo.compute_loss

    def _spy(self, policy, ref_policy, batch):
        seen_batches.append(
            [
                (
                    int(rec.metadata["turn_index"]),
                    float(rec.reward),
                    str(rec.metadata["turn_credit"]["mode"]),
                    str(rec.metadata["prompt_group_id"]),
                    str(rec.metadata["turn_group_id"]),
                )
                for rec in batch.records
            ]
        )
        return original(policy, ref_policy, batch)

    trainer.algo.compute_loss = MethodType(_spy, trainer.algo)
    stats = trainer.train()

    assert seen_batches
    first_batch = seen_batches[0]
    assert len(first_batch) == 4
    assert all(mode == "hybrid" for _turn, _reward, mode, _prompt_gid, _turn_gid in first_batch)
    assert all(
        prompt_gid == "calc-1" for _turn, _reward, _mode, prompt_gid, _turn_gid in first_batch
    )
    assert first_batch[0][0] == 0 and first_batch[1][0] == 1
    assert first_batch[0][1] > first_batch[1][1]
    assert first_batch[2][1] > first_batch[3][1]
    assert first_batch[0][4] == "calc-1::turn:0"
    assert first_batch[1][4] == "calc-1::turn:1"
    assert first_batch[2][4] == "calc-1::turn:0"
    assert first_batch[3][4] == "calc-1::turn:1"
    assert stats.iters[0]["turn_credit_local_component_mean"] > 0.0
    assert stats.iters[0]["turn_credit_reward_mean"] > 0.0


def test_multi_turn_credit_config_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        parse_multi_turn_credit_config({"mode": "hybrid", "gamma": -0.1})


def test_turn_specific_grouping_avoids_cross_turn_grpo_bias():
    shared_adv = group_normalize_advantage([0.93, 0.70, 0.93, 0.70], eps=1e-6)
    split_adv = group_normalize_advantage([0.93, 0.93], eps=1e-6) + group_normalize_advantage(
        [0.70, 0.70], eps=1e-6
    )

    assert shared_adv[0] > 0.0
    assert shared_adv[1] < 0.0
    assert split_adv == [0.0, 0.0, 0.0, 0.0]


def test_sim_tool_env_builds_teacher_turn_samples():
    item = build_sim_tool_dataset(n=1, seed=0)[0]
    env = SimToolEnv([item])

    samples = env.build_supervised_samples(item)

    assert len(samples) == 2
    assert samples[0].response == f"<tool_call>calc({item['expr']})</tool_call>"
    assert samples[0].metadata["turn_index"] == 0
    assert "<tool_result>" in samples[1].prompt_suffix
    assert item["target"] in samples[1].prompt_suffix
    assert samples[1].response == f"answer={item['target']}"
    assert samples[1].metadata["turn_index"] == 1


def test_grpo_trainer_runs_interleaved_sft_when_enabled():
    env = EchoTaskEnv([{"task_id": "echo-1", "instruction": "Say: hello", "target": "hello"}])
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    trainer = GRPOTrainer(
        policy=_tiny(),
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=2,
            group_size=1,
            prompts_per_iter=1,
            lr=1e-3,
            max_new_tokens=4,
            temperature=1.0,
            log_every=100,
            seed=0,
            interleave_sft_every=1,
            interleave_sft_samples=2,
            interleave_sft_lr=1e-3,
        ),
    )

    stats = trainer.train()

    assert len(stats.iters) == 2
    assert "sft_loss" not in stats.iters[0]
    assert stats.iters[1]["sft_loss"] > 0.0
    assert stats.iters[1]["n_sft_samples"] == 2
    assert stats.iters[1]["n_sft_steps"] >= 1


def test_grpo_trainer_bootstrap_sft_improves_first_rollout_reward():
    env = EchoTaskEnv([{"task_id": "echo-1", "instruction": "Say: hello", "target": "hello"}])
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    trainer = GRPOTrainer(
        policy=_tiny(),
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=1,
            prompts_per_iter=1,
            lr=1e-3,
            max_new_tokens=8,
            temperature=0.0,
            log_every=100,
            seed=0,
            bootstrap_sft_rounds=10,
            bootstrap_sft_samples=8,
            bootstrap_sft_lr=5e-2,
        ),
    )

    stats = trainer.train()

    assert len(stats.iters) == 2
    assert stats.iters[0]["algo"] == "sft_bootstrap"
    assert stats.iters[0]["sft_loss"] > 0.0
    assert stats.iters[1]["algo"] == "grpo"
    assert stats.iters[1]["mean_reward"] > 0.0


def test_sim_tool_reward_when_answer_matches():
    dataset = build_sim_tool_dataset(n=4, seed=0)
    env = SimToolEnv(dataset)
    rm = RewardManager([SimToolRewardComponent(weight=1.0)])

    async def _run():
        await env.setup()
        item = await env.get_next_item()
        # fabricate a "perfect" trajectory by using the env's expected answer
        from hermes_agentic_rl.core.types import RolloutStep, Trajectory

        traj = Trajectory(
            task_id=item["task_id"],
            prompt=item["instruction"],
            steps=[
                RolloutStep(
                    turn_index=0,
                    tool_calls=[{"name": "calc", "arg": item["expr"]}],
                    tool_results=[{"name": "calc", "result": item["target"]}],
                )
            ],
            final_output=f"answer={item['target']}",
            finished_naturally=True,
            turns_used=1,
            metadata={},
        )
        return await rm.evaluate(item, traj, tool_context=None)

    summary = asyncio.run(_run())
    assert summary.final_score == pytest.approx(1.0, abs=1e-6)


def test_curriculum_promotes_once_threshold_met():
    l0 = EchoTaskEnv(build_default_echo_dataset())
    l1 = SimToolEnv(build_sim_tool_dataset(n=4, seed=0))
    changes: list[tuple[int, int]] = []
    env = CurriculumEnv(
        levels=[l0, l1],
        window=4,
        promote_threshold=0.5,
        on_level_change=lambda a, b: changes.append((a, b)),
    )
    assert env.current_level == 0
    for _ in range(4):
        env.observe(0.8)
    assert env.current_level == 1
    assert changes == [(0, 1)]
    snap = env.snapshot()
    assert snap["promotions"] == 1


def test_curriculum_demote_optional():
    l0 = EchoTaskEnv(build_default_echo_dataset())
    l1 = SimToolEnv(build_sim_tool_dataset(n=4, seed=0))
    env = CurriculumEnv(
        levels=[l0, l1],
        window=4,
        promote_threshold=0.5,
        allow_demote=True,
        demote_threshold=0.1,
    )
    # promote
    for _ in range(4):
        env.observe(0.9)
    assert env.current_level == 1
    # demote
    for _ in range(4):
        env.observe(0.0)
    assert env.current_level == 0
