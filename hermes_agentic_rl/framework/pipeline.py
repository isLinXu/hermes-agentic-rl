from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hermes_agentic_rl.collectors.preference_mining import (
    build_preference_pairs_from_samples,
    records_to_train_samples,
)
from hermes_agentic_rl.collectors.replay_export import (
    build_tokenizer_from_config,
    record_to_session_turn_samples,
    replay_jsonl_payloads_from_record,
)
from hermes_agentic_rl.collectors.session_judge import judge_session_turn_sample
from hermes_agentic_rl.collectors.trajectory_adapter import (
    session_turn_sample_to_train_sample,
    trajectory_to_session_turn_samples,
)
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.offline.replay_buffer import DPOPair, ReplayBuffer, TrainSample
from hermes_agentic_rl.rewards.base import BaseReward
from hermes_agentic_rl.rewards.filesystem_verifier_reward import (
    FileSystemVerifierReward,
)
from hermes_agentic_rl.rewards.next_turn_feedback import NextTurnFeedbackReward
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter
from hermes_agentic_rl.trainers.base import BaseTrainer

if TYPE_CHECKING:
    from hermes_agentic_rl.collectors.sidecar import LocalSessionSidecar


def _build_runtime_adapter(config: dict[str, Any]) -> BaseRuntimeAdapter:
    integration = str(config.get("runtime", {}).get("integration", "fake"))
    if integration == "fake":
        from hermes_agentic_rl.runtime.fake_adapter import FakeRuntimeAdapter

        return FakeRuntimeAdapter()
    if integration == "hermes":
        from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter

        return HermesRuntimeAdapter()
    raise ValueError(f"unsupported runtime.integration: {integration!r}")


def _build_reward_manager(config: dict[str, Any]) -> RewardManager:
    reward_cfg = config.get("reward", {}) or {}
    component_specs = reward_cfg.get("components")
    if not isinstance(component_specs, list) or not component_specs:
        component_specs = [
            {"name": "outcome_reward", "weight": 0.2},
            {"name": "toolcall_reward", "weight": 0.2},
            {"name": "filesystem_verifier_reward", "weight": 0.6},
        ]

    rewards: list[BaseReward] = []
    for spec in component_specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        weight = float(spec.get("weight", 1.0))
        if name == "outcome_reward":
            rewards.append(OutcomeReward(weight=weight))
        elif name == "toolcall_reward":
            rewards.append(ToolcallReward(weight=weight))
        elif name == "filesystem_verifier_reward":
            rewards.append(FileSystemVerifierReward(weight=weight))
        elif name == "next_turn_feedback_reward":
            rewards.append(NextTurnFeedbackReward(weight=weight))

    return RewardManager(rewards=rewards)


def _prompt_from_item(item: dict[str, Any]) -> str:
    for key in ("instruction", "prompt", "task", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return str(item.get("task_id", ""))


@dataclass(slots=True)
class EnvTrainingPipeline:
    config: dict[str, Any]
    runtime_adapter: BaseRuntimeAdapter
    reward_manager: RewardManager
    trainer_bridge: TrainerBridge | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        trainer: BaseTrainer | None = None,
        runtime_adapter: BaseRuntimeAdapter | None = None,
    ) -> EnvTrainingPipeline:
        adapter = runtime_adapter or _build_runtime_adapter(config)
        bridge = TrainerBridge(trainer) if trainer is not None else None
        return cls(
            config=dict(config),
            runtime_adapter=adapter,
            reward_manager=_build_reward_manager(config),
            trainer_bridge=bridge,
        )

    def build_agent_loop(self) -> Any:
        return self.runtime_adapter.build_agent_loop(self.config)

    def build_rollout_manager(self) -> RolloutManager:
        return RolloutManager(self.build_agent_loop())

    async def rollout(self, item: dict[str, Any], prompt: str | None = None) -> Trajectory:
        rollout_prompt = prompt or _prompt_from_item(item)
        return await self.build_rollout_manager().collect(item, rollout_prompt)

    async def judge(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any = None,
    ) -> RewardSummary:
        return await self.reward_manager.evaluate(item, trajectory, tool_context)

    async def collect_and_judge(
        self,
        item: dict[str, Any],
        prompt: str | None = None,
        tool_context: Any = None,
    ) -> tuple[Trajectory, RewardSummary]:
        trajectory = await self.rollout(item, prompt=prompt)
        summary = await self.judge(item, trajectory, tool_context=tool_context)
        return trajectory, summary

    async def export(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        if self.trainer_bridge is None:
            raise RuntimeError("trainer_bridge is not configured")
        return await self.trainer_bridge.submit(item, trajectory, reward_summary)

    def describe(self) -> dict[str, Any]:
        reward_names = [type(reward).__name__ for reward in self.reward_manager.rewards]
        return {
            "kind": "env",
            "runtime_integration": str(self.config.get("runtime", {}).get("integration", "fake")),
            "runtime_available": bool(self.runtime_adapter.is_available()),
            "reward_components": reward_names,
            "has_trainer_bridge": self.trainer_bridge is not None,
        }


@dataclass(slots=True)
class SessionTrainingPipeline:
    config: dict[str, Any]
    tokenizer: Any
    judge_config: dict[str, Any] = field(default_factory=dict, repr=False)
    session_sidecar: LocalSessionSidecar | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        build_sidecar: bool = True,
    ) -> SessionTrainingPipeline:
        runtime_cfg = dict(config.get("runtime", {}) or {})
        session_sidecar_cfg = runtime_cfg.get("session_sidecar")
        judge_cfg: dict[str, Any] = {}
        if isinstance(config.get("judge"), dict):
            judge_cfg = dict(config["judge"])
        if isinstance(session_sidecar_cfg, dict) and session_sidecar_cfg.get("judge"):
            judge_cfg = dict(session_sidecar_cfg.get("judge") or {})

        sidecar = None
        if build_sidecar:
            from hermes_agentic_rl.collectors.sidecar import (
                build_sidecar_from_runtime_config,
            )

            sidecar = build_sidecar_from_runtime_config(runtime_cfg)
        return cls(
            config=dict(config),
            tokenizer=build_tokenizer_from_config(config),
            judge_config=judge_cfg,
            session_sidecar=sidecar,
        )

    def submit_record_to_sidecar(self, record: dict[str, Any]) -> bool:
        if self.session_sidecar is None:
            raise RuntimeError("session_sidecar is not configured")
        return bool(self.session_sidecar.submit(record))

    def submit_trajectory_to_sidecar(self, trajectory: Trajectory) -> bool:
        return self.submit_record_to_sidecar(trajectory_to_dict(trajectory))

    def flush_sidecar(self, timeout: float | None = None) -> None:
        if self.session_sidecar is None:
            raise RuntimeError("session_sidecar is not configured")
        self.session_sidecar.flush(timeout=timeout)

    def close_sidecar(self) -> None:
        if self.session_sidecar is None:
            return
        self.session_sidecar.close()

    def samples_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        task_id: str | None = None,
    ) -> list[Any]:
        from hermes_agentic_rl.collectors.conversation_collector import (
            collect_session_turn_samples,
        )

        return collect_session_turn_samples(messages, session_id=session_id, task_id=task_id)

    def samples_from_trajectory(self, trajectory: Trajectory) -> list[Any]:
        return trajectory_to_session_turn_samples(trajectory)

    def samples_from_record(self, record: Any) -> list[Any]:
        return record_to_session_turn_samples(record)

    def train_samples_from_record(self, record: Any) -> list[TrainSample]:
        train_samples: list[TrainSample] = []
        for sample in self.samples_from_record(record):
            summary = judge_session_turn_sample(sample, self.judge_config)
            train_samples.append(
                session_turn_sample_to_train_sample(
                    sample,
                    tokenizer=self.tokenizer,
                    reward_summary=summary,
                )
            )
        return train_samples

    def train_samples_from_records(
        self,
        records: list[Any],
        *,
        min_reward: float = -1e9,
    ) -> list[TrainSample]:
        train_samples: list[TrainSample] = []
        for record in records:
            for sample in self.samples_from_record(record):
                summary = judge_session_turn_sample(sample, self.judge_config)
                train_sample = session_turn_sample_to_train_sample(
                    sample,
                    tokenizer=self.tokenizer,
                    reward_summary=summary,
                )
                if float(train_sample.reward) >= float(min_reward):
                    train_samples.append(train_sample)
        return train_samples

    def replay_buffer_from_record(self, record: Any) -> ReplayBuffer:
        return ReplayBuffer.from_samples(self.train_samples_from_record(record))

    def replay_buffer_from_records(
        self,
        records: list[Any],
        *,
        min_reward: float = -1e9,
    ) -> ReplayBuffer:
        return ReplayBuffer.from_samples(
            self.train_samples_from_records(records, min_reward=min_reward)
        )

    def replay_train_samples_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        min_reward: float = -1e9,
    ) -> list[TrainSample]:
        return list(records_to_train_samples(records, min_reward=min_reward))

    def replay_buffer_from_replay_records(
        self,
        records: list[dict[str, Any]],
        *,
        min_reward: float = -1e9,
    ) -> ReplayBuffer:
        return ReplayBuffer.from_samples(
            self.replay_train_samples_from_records(records, min_reward=min_reward)
        )

    def preference_pairs_from_train_samples(
        self,
        samples: list[TrainSample],
        *,
        min_reward_gap: float = 0.25,
        max_pairs_per_prompt: int = 1,
    ) -> list[DPOPair]:
        return build_preference_pairs_from_samples(
            samples,
            min_reward_gap=min_reward_gap,
            max_pairs_per_prompt=max_pairs_per_prompt,
        )

    def preference_pairs_from_replay_records(
        self,
        records: list[dict[str, Any]],
        *,
        min_reward: float = -1e9,
        min_reward_gap: float = 0.25,
        max_pairs_per_prompt: int = 1,
    ) -> list[DPOPair]:
        samples = self.replay_train_samples_from_records(records, min_reward=min_reward)
        return self.preference_pairs_from_train_samples(
            samples,
            min_reward_gap=min_reward_gap,
            max_pairs_per_prompt=max_pairs_per_prompt,
        )

    def preference_pairs_from_records(
        self,
        records: list[Any],
        *,
        min_reward: float = -1e9,
        min_reward_gap: float = 0.25,
        max_pairs_per_prompt: int = 1,
    ) -> list[DPOPair]:
        samples = self.train_samples_from_records(records, min_reward=min_reward)
        return self.preference_pairs_from_train_samples(
            samples,
            min_reward_gap=min_reward_gap,
            max_pairs_per_prompt=max_pairs_per_prompt,
        )

    def replay_payloads_from_record(self, record: Any) -> list[dict[str, Any]]:
        return replay_jsonl_payloads_from_record(
            record,
            tokenizer=self.tokenizer,
            judge_config=self.judge_config,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "session",
            "tokenizer": type(self.tokenizer).__name__,
            "has_sidecar": self.session_sidecar is not None,
            "judge_components": [
                str(spec.get("name"))
                for spec in (self.judge_config.get("components") or [])
                if isinstance(spec, dict)
            ],
        }


@dataclass(slots=True)
class HermesAgenticRLFramework:
    config: dict[str, Any]
    env: EnvTrainingPipeline
    session: SessionTrainingPipeline

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        trainer: BaseTrainer | None = None,
        runtime_adapter: BaseRuntimeAdapter | None = None,
        build_sidecar: bool = True,
    ) -> HermesAgenticRLFramework:
        return cls(
            config=dict(config),
            env=EnvTrainingPipeline.from_config(
                config,
                trainer=trainer,
                runtime_adapter=runtime_adapter,
            ),
            session=SessionTrainingPipeline.from_config(
                config,
                build_sidecar=build_sidecar,
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "hermes-agentic-rl",
            "env": self.env.describe(),
            "session": self.session.describe(),
        }


def build_framework(
    config: dict[str, Any],
    *,
    trainer: BaseTrainer | None = None,
    runtime_adapter: BaseRuntimeAdapter | None = None,
    build_sidecar: bool = True,
) -> HermesAgenticRLFramework:
    return HermesAgenticRLFramework.from_config(
        config,
        trainer=trainer,
        runtime_adapter=runtime_adapter,
        build_sidecar=build_sidecar,
    )
