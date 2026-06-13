"""`train-rl` CLI: real RL training on a configured env.

Supports:
  - algo: "grpo" (default) | "ppo" | "hybrid" (GRPO + OPD, OpenClaw-RL §3.3)
  - env: "echo" | "sim_tool" | "curriculum" | "letter_counting" | "hermes_reasoning_traces"
  - backend: "tiny" | "hf" (HuggingFace AutoModelForCausalLM)
  - agent_loop: "policy" (default, single-turn) | "multi_turn" (tool-use)
  - optional live dashboard (pure stdlib HTTP)
  - v0.5: per-token advantage, interleaved SFT, batch generate

The existing v0.2 config schema (echo + grpo) is 100% preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.agent_loop.multi_turn_loop import MultiTurnAgentLoop
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.envs.context_benchmark import ContextBenchmarkEnv
from hermes_agentic_rl.envs.curriculum import CurriculumEnv
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.envs.hermes_reasoning_traces import (
    HermesReasoningTraceEnv,
)
from hermes_agentic_rl.envs.sim_tool_env import (
    DEFAULT_TOOLS,
    SimToolEnv,
    SimToolRewardComponent,
    build_sim_tool_dataset,
)
from hermes_agentic_rl.monitor.dashboard import LiveDashboard
from hermes_agentic_rl.monitor.writers import (
    MultiMetricsWriter,
    build_writer_from_config,
)
from hermes_agentic_rl.rewards.base import BaseReward
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from hermes_agentic_rl.trainers.hybrid_trainer import (
    HybridTrainer,
    HybridTrainerConfig,
)
from hermes_agentic_rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig

KLEstimator = Literal["k1", "k2", "k3"]


def _kl_estimator(value: Any) -> KLEstimator:
    candidate = str(value or "k1")
    if candidate in {"k1", "k2", "k3"}:
        return cast(KLEstimator, candidate)
    raise ValueError("kl_estimator must be one of: k1, k2, k3")


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _profile_output_path(tcfg: dict[str, Any], output_dir: Path | None) -> Path | None:
    explicit = _optional_path(tcfg.get("profile_output_path"))
    if explicit is not None:
        return explicit
    if bool(tcfg.get("profile", False)) and output_dir is not None:
        return output_dir / "profile.jsonl"
    return None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------


def _build_backend(cfg: dict[str, Any], *, need_value_head: bool) -> LLMBackend:
    backend_cfg = cfg.get("backend", {}) or {}
    name = backend_cfg.get("name", "tiny")
    if name == "tiny":
        return TinyCausalLMBackend(
            TinyBackendConfig(
                dim=int(backend_cfg.get("dim", 32)),
                n_heads=int(backend_cfg.get("n_heads", 4)),
                n_layers=int(backend_cfg.get("n_layers", 2)),
                max_len=int(backend_cfg.get("max_len", 128)),
                device=str(backend_cfg.get("device", "cpu")),
                dtype=str(backend_cfg.get("dtype", "float32")),
                seed=backend_cfg.get("seed", 0),
                with_value_head=need_value_head or bool(backend_cfg.get("with_value_head", False)),
                use_sdpa=bool(
                    backend_cfg.get("use_sdpa", backend_cfg.get("flash_attention", False))
                ),
            )
        )
    if name == "hf":
        from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend

        return HFCausalLMBackend(
            HFBackendConfig(
                model_name_or_path=str(backend_cfg.get("model_name_or_path", "gpt2")),
                device=str(backend_cfg.get("device", "cpu")),
                dtype=str(backend_cfg.get("dtype", "float32")),
                with_value_head=need_value_head or bool(backend_cfg.get("with_value_head", False)),
                trust_remote_code=bool(backend_cfg.get("trust_remote_code", False)),
                flash_attention=bool(backend_cfg.get("flash_attention", False)),
                extra_model_kwargs=(
                    dict(backend_cfg.get("extra_model_kwargs") or {})
                    if isinstance(backend_cfg.get("extra_model_kwargs") or {}, dict)
                    else {}
                ),
            )
        )
    raise RuntimeError(f"backend '{name}' not supported; choose 'tiny' or 'hf'.")


def _backend_name(cfg: dict[str, Any]) -> str:
    backend_cfg = cfg.get("backend", {}) or {}
    return str(backend_cfg.get("name", "tiny"))


# ----------------------------------------------------------------------------
# Env + rewards
# ----------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_single_env(env_cfg: dict[str, Any]) -> tuple[BaseEnv, RewardManager]:
    env_type = env_cfg.get("type", "echo")
    if env_type == "echo":
        dataset_path = env_cfg.get("dataset_path")
        dataset = _load_jsonl(dataset_path) if dataset_path else build_default_echo_dataset()
        return EchoTaskEnv(dataset), RewardManager([EchoRewardComponent(weight=1.0)])
    if env_type == "sim_tool":
        n = int(env_cfg.get("dataset_size", 16))
        seed = int(env_cfg.get("dataset_seed", 0))
        dataset = build_sim_tool_dataset(n=n, seed=seed)
        return SimToolEnv(dataset), RewardManager([SimToolRewardComponent(weight=1.0)])
    if env_type == "letter_counting":
        from hermes_agentic_rl.envs.letter_counting import (
            LetterCountingConfig,
            LetterCountingEnv,
            LetterCountingReward,
        )

        seed = int(env_cfg.get("dataset_seed", 42))
        return (
            LetterCountingEnv(
                LetterCountingConfig(
                    max_level=int(env_cfg.get("max_level", 10)),
                    seed=seed,
                )
            ),
            RewardManager([LetterCountingReward(weight=1.0)]),
        )
    if env_type == "hermes_reasoning_traces":
        hermes_env = HermesReasoningTraceEnv.from_hf_dataset(env_cfg)
        return hermes_env, RewardManager([hermes_env.reward_component])
    if env_type == "context_benchmark":
        context_env = ContextBenchmarkEnv.from_config(env_cfg)
        return context_env, RewardManager([context_env.reward_component])
    raise RuntimeError(f"env type '{env_type}' not supported")


def _build_reward_component(spec: dict[str, Any]) -> BaseReward:
    """Instantiate a reward component from a YAML component spec."""
    typ = str(spec.get("type") or spec.get("name") or "").strip()
    weight = float(spec.get("weight", 1.0))
    if typ in {"LengthPenaltyReward", "length_penalty"}:
        from hermes_agentic_rl.rewards.length_penalty import (
            LengthPenaltyConfig,
            LengthPenaltyReward,
        )

        cfg = LengthPenaltyConfig(
            target_len=int(spec.get("target_len", 512)),
            alpha=float(spec.get("alpha", 0.1)),
            mode=str(spec.get("mode", "linear")),
            apply_on=str(spec.get("apply_on", "response")),
        )
        return LengthPenaltyReward(cfg, weight=weight)
    if typ in {"letter_counting", "LetterCountingReward"}:
        from hermes_agentic_rl.envs.letter_counting import LetterCountingReward

        return LetterCountingReward(weight=weight)
    if typ in {"letter_counting_next_state", "LetterCountingNextStateReward"}:
        from hermes_agentic_rl.envs.letter_counting import (
            LetterCountingNextStateReward,
        )

        return LetterCountingNextStateReward(
            weight=weight,
            emit_hint_on_correct=bool(spec.get("emit_hint_on_correct", False)),
        )
    raise RuntimeError(f"reward component type '{typ}' not supported")


def _build_reward_manager(
    cfg: dict[str, Any],
    *,
    env_cfg: dict[str, Any],
    default_manager: RewardManager,
) -> RewardManager:
    reward_cfg = cfg.get("reward") or {}
    components = reward_cfg.get("components")
    if not components:
        return default_manager
    built = [_build_reward_component(dict(spec)) for spec in components]
    return RewardManager(built)


def _build_reward_model_component(cfg: dict[str, Any]) -> Any | None:
    """Build a RewardModelComponent from the `reward_model:` YAML section.

    Accepted keys::

        reward_model:
          enabled: true
          backend: tiny | hf    # default tiny
          # tiny backend options (standalone RM — not the RL policy)
          dim: 32
          n_heads: 4
          n_layers: 2
          max_len: 256
          seed: 0
          # hf backend options
          model_name_or_path: gpt2
          device: cpu
          dtype: float32
          trust_remote_code: false
          # Pre-trained RM head checkpoint (from RewardModelTrainer.save_head).
          head_path: path/to/rm_head.pt
          weight: 1.0
          freeze_base: true     # default True
    """
    rm_cfg = cfg.get("reward_model") or {}
    if not rm_cfg or not bool(rm_cfg.get("enabled", False)):
        return None

    # lazy import — RM pulls in torch, so keep it behind the opt-in flag.
    import torch

    from hermes_agentic_rl.rewards.reward_model import (
        RewardModel,
        RewardModelComponent,
    )

    backend_name = str(rm_cfg.get("backend", "tiny"))
    backend: LLMBackend
    if backend_name == "tiny":
        backend = TinyCausalLMBackend(
            TinyBackendConfig(
                dim=int(rm_cfg.get("dim", 32)),
                n_heads=int(rm_cfg.get("n_heads", 4)),
                n_layers=int(rm_cfg.get("n_layers", 2)),
                max_len=int(rm_cfg.get("max_len", 256)),
                seed=rm_cfg.get("seed", 0),
                with_value_head=False,
            )
        )
    elif backend_name == "hf":
        from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend

        backend = HFCausalLMBackend(
            HFBackendConfig(
                model_name_or_path=str(rm_cfg.get("model_name_or_path", "gpt2")),
                device=str(rm_cfg.get("device", "cpu")),
                dtype=str(rm_cfg.get("dtype", "float32")),
                with_value_head=False,
                trust_remote_code=bool(rm_cfg.get("trust_remote_code", False)),
            )
        )
    else:
        raise RuntimeError(f"reward_model.backend '{backend_name}' not supported")

    rm = RewardModel(backend, freeze_base=bool(rm_cfg.get("freeze_base", True)))

    head_path = rm_cfg.get("head_path")
    if head_path:
        head_state = torch.load(str(head_path), map_location="cpu", weights_only=True)
        rm.head.load_state_dict(head_state)
        print(f"[train-rl] reward_model: loaded RM head from {head_path}")
    else:
        print("[train-rl] reward_model: WARNING no head_path; using randomly initialized head")

    rm.eval()
    return RewardModelComponent(rm, weight=float(rm_cfg.get("weight", 1.0)))


def _build_env_and_rewards(cfg: dict[str, Any]) -> tuple[BaseEnv, RewardManager]:
    env_cfg = cfg.get("environment", {}) or {}
    env_type = env_cfg.get("type", "echo")
    if env_type not in {"curriculum", "multi_stream"}:
        env, rm_manager = _build_single_env(env_cfg)
        rm_manager = _build_reward_manager(cfg, env_cfg=env_cfg, default_manager=rm_manager)
        extra = _build_reward_model_component(cfg)
        if extra is not None:
            rm_manager.rewards.append(extra)
        return env, rm_manager

    if env_type == "multi_stream":
        return _build_multi_stream(env_cfg)

    sub_specs = env_cfg.get("levels") or []
    if not sub_specs:
        raise RuntimeError("curriculum env requires non-empty 'levels' list")
    sub_envs: list[BaseEnv] = []
    reward_managers: list[RewardManager] = []
    for spec in sub_specs:
        sub_env, sub_rm = _build_single_env(dict(spec))
        sub_envs.append(sub_env)
        reward_managers.append(sub_rm)
    cur_env = CurriculumEnv(
        levels=sub_envs,
        window=int(env_cfg.get("window", 20)),
        promote_threshold=float(env_cfg.get("promote_threshold", 0.6)),
        demote_threshold=env_cfg.get("demote_threshold"),
        allow_demote=bool(env_cfg.get("allow_demote", False)),
        on_level_change=lambda old, new: print(f"[curriculum] level {old} -> {new}"),
    )

    # Reward dispatches through env.compute_reward → active.compute_reward;
    # we build a thin aggregator that asks the current level's reward manager.
    class _CurrRewardManager:
        async def evaluate(self, item, trajectory, tool_context):  # type: ignore[no-untyped-def]
            lvl = cur_env.current_level
            return await reward_managers[lvl].evaluate(item, trajectory, tool_context)

    return cur_env, _CurrRewardManager()  # type: ignore[return-value]


def _build_multi_stream(env_cfg: dict[str, Any]) -> tuple[BaseEnv, RewardManager]:
    """Build a :class:`MixedCurriculumEnv` from a ``multi_stream`` env spec.

    YAML shape::

        environment:
          type: multi_stream
          window: 20
          adapt_lr: 0.1
          min_weight: 0.01
          target_reward: 0.5
          streams:
            - {type: echo, weight: 1.0}
            - {type: letter_counting, weight: 2.0}

    Each stream carries its own reward manager; rewards are routed per-rollout
    by the ``_curriculum_level`` tag the mixture stamps onto every item.
    """
    from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv

    sub_specs = env_cfg.get("streams") or env_cfg.get("levels") or []
    if not sub_specs:
        raise RuntimeError("multi_stream env requires a non-empty 'streams' list")
    sub_envs: list[BaseEnv] = []
    reward_managers: list[RewardManager] = []
    weights: list[float] = []
    for spec in sub_specs:
        spec = dict(spec)
        weights.append(float(spec.pop("weight", 1.0)))
        sub_env, sub_rm = _build_single_env(spec)
        sub_envs.append(sub_env)
        reward_managers.append(sub_rm)

    mixed = MixedCurriculumEnv(
        levels=sub_envs,
        weights=weights,
        window=int(env_cfg.get("window", 20)),
        adapt_lr=float(env_cfg.get("adapt_lr", 0.1)),
        min_weight=float(env_cfg.get("min_weight", 0.01)),
        target_reward=float(env_cfg.get("target_reward", 0.5)),
        seed=env_cfg.get("seed", 0),
    )

    # Route reward by the per-item stream tag (robust to mixed sampling).
    class _MixedRewardManager:
        async def evaluate(self, item, trajectory, tool_context):  # type: ignore[no-untyped-def]
            lvl = int(item.get("_curriculum_level", mixed.current_level))
            lvl = lvl if 0 <= lvl < len(reward_managers) else mixed.current_level
            return await reward_managers[lvl].evaluate(item, trajectory, tool_context)

    return mixed, _MixedRewardManager()  # type: ignore[return-value]


# ----------------------------------------------------------------------------
# Agent-loop factory
# ----------------------------------------------------------------------------


def _make_agent_loop_factory(cfg: dict[str, Any]):
    loop_cfg = cfg.get("agent_loop", {}) or {}
    kind = loop_cfg.get("type", "policy")
    max_turns = int(loop_cfg.get("max_turns", 3))
    stop_strings = list(loop_cfg.get("stop_strings") or [])
    if kind == "policy":
        tcfg = cfg.get("train_rl", {}) or {}
        temp = float(tcfg.get("temperature", 1.0))
        max_new_tokens = int(tcfg.get("max_new_tokens", 16))

        def _factory(*, backend: LLMBackend, seed: int | None):
            from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop

            return PolicyAgentLoop(
                backend=backend,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                seed=seed,
                stop_strings=stop_strings,
            )

        return _factory
    if kind == "multi_turn":
        tcfg = cfg.get("train_rl", {}) or {}
        mnt_per_turn = int(loop_cfg.get("max_new_tokens_per_turn", tcfg.get("max_new_tokens", 16)))
        temp = float(tcfg.get("temperature", 1.0))

        def _factory(*, backend: LLMBackend, seed: int | None):
            return MultiTurnAgentLoop(
                backend=backend,
                tools=DEFAULT_TOOLS,
                max_turns=max_turns,
                max_new_tokens_per_turn=mnt_per_turn,
                temperature=temp,
                seed=seed,
                stop_strings=stop_strings,
            )

        return _factory
    raise RuntimeError(f"agent_loop type '{kind}' not supported")


# ----------------------------------------------------------------------------
# Trainer construction
# ----------------------------------------------------------------------------


def _build_grpo(cfg: dict[str, Any], output_dir: str | None) -> GRPOTrainerConfig:
    tcfg = cfg.get("train_rl", {}) or {}
    out = _optional_path(output_dir if output_dir else tcfg.get("output_dir"))
    return GRPOTrainerConfig(
        n_iters=int(tcfg.get("n_iters", 30)),
        group_size=int(tcfg.get("group_size", 4)),
        prompts_per_iter=int(tcfg.get("prompts_per_iter", 2)),
        lr=float(tcfg.get("lr", 5e-3)),
        max_new_tokens=int(tcfg.get("max_new_tokens", 8)),
        temperature=float(tcfg.get("temperature", 1.0)),
        clip_eps=float(tcfg.get("clip_eps", 0.2)),
        kl_coef=float(tcfg.get("kl_coef", 0.0)),
        entropy_coef=float(tcfg.get("entropy_coef", 0.0)),
        use_reference=bool(tcfg.get("use_reference", False)),
        log_every=int(tcfg.get("log_every", 5)),
        log_format=str(tcfg.get("log_format", "text")),
        save_every=int(tcfg.get("save_every", 0)),
        output_dir=out,
        seed=tcfg.get("seed", 0),
        multi_turn=bool(tcfg.get("multi_turn", False)),
        multi_turn_credit=_optional_dict(tcfg.get("multi_turn_credit")),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        profile=bool(tcfg.get("profile", False)),
        profile_output_path=_profile_output_path(tcfg, out),
        update_epochs=int(tcfg.get("update_epochs", 1)),
        minibatch_size=int(tcfg.get("minibatch_size", 0)),
        shuffle_minibatches=bool(tcfg.get("shuffle_minibatches", True)),
        loss_agg=tcfg.get("loss_agg", "mean_token"),
        max_len_for_dr_grpo=int(tcfg.get("max_len_for_dr_grpo", 256)),
        advantage_eps=float(tcfg.get("advantage_eps", 1e-6)),
        # v0.5: per-token advantage + interleaved SFT
        per_token_advantage=bool(tcfg.get("per_token_advantage", False)),
        advantage_norm=tcfg.get("advantage_norm", "group"),
        interleave_sft_every=int(tcfg.get("interleave_sft_every", 0)),
        interleave_sft_samples=int(tcfg.get("interleave_sft_samples", 32)),
        interleave_sft_lr=float(tcfg.get("interleave_sft_lr", 1e-4)),
        interleave_sft_epochs=int(tcfg.get("interleave_sft_epochs", 1)),
        interleave_sft_batch_size=int(tcfg.get("interleave_sft_batch_size", 8)),
        bootstrap_sft_rounds=int(tcfg.get("bootstrap_sft_rounds", 0)),
        bootstrap_sft_samples=int(tcfg.get("bootstrap_sft_samples", 32)),
        bootstrap_sft_lr=float(tcfg.get("bootstrap_sft_lr", 1e-4)),
        bootstrap_sft_epochs=int(tcfg.get("bootstrap_sft_epochs", 1)),
        # v0.5: batch generate
        batch_generate=bool(tcfg.get("batch_generate", False)),
        # v0.6: checkpoint / resume
        checkpoint_every=int(tcfg.get("checkpoint_every", 0)),
        keep_last_checkpoints=int(tcfg.get("keep_last_checkpoints", 3)),
        resume_from=tcfg.get("resume_from"),
        auto_resume=bool(tcfg.get("auto_resume", False)),
        # v0.6: KL estimator
        kl_estimator=_kl_estimator(tcfg.get("kl_estimator", "k1")),
        # v0.7: best checkpoint + early stop
        save_best_checkpoint=bool(tcfg.get("save_best_checkpoint", False)),
        early_stop_patience=int(tcfg.get("early_stop_patience", 0)),
        early_stop_min_delta=float(tcfg.get("early_stop_min_delta", 1e-4)),
        target_kl=float(tcfg.get("target_kl", 0.0)),
        adaptive_kl=bool(tcfg.get("adaptive_kl", False)),
        adaptive_kl_horizon=float(tcfg.get("adaptive_kl_horizon", 10000.0)),
        adaptive_kl_min=float(tcfg.get("adaptive_kl_min", 1e-4)),
        adaptive_kl_max=float(tcfg.get("adaptive_kl_max", 10.0)),
        normalize_reward=bool(tcfg.get("normalize_reward", False)),
        reward_norm_clip=float(tcfg.get("reward_norm_clip", 10.0)),
        amp_dtype=str(tcfg.get("amp_dtype", "fp32")),
        grad_accum_steps=int(tcfg.get("grad_accum_steps", 1)),
        vllm_rollout_model=tcfg.get("vllm_rollout_model") or None,
        vllm_tensor_parallel_size=int(tcfg.get("vllm_tensor_parallel_size", 1)),
        vllm_max_model_len=int(tcfg.get("vllm_max_model_len", 4096)),
        vllm_gpu_memory_utilization=float(tcfg.get("vllm_gpu_memory_utilization", 0.90)),
        vllm_enable_prefix_caching=bool(tcfg.get("vllm_enable_prefix_caching", True)),
        vllm_sync_every=int(tcfg.get("vllm_sync_every", 1)),
        distributed_strategy=str(tcfg.get("distributed_strategy", "none")),
        fsdp_cpu_offload=bool(tcfg.get("fsdp_cpu_offload", False)),
        flash_attention=bool(tcfg.get("flash_attention", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        replay_buffer=_optional_dict(tcfg.get("replay_buffer") or cfg.get("replay_buffer")),
        replay_mix_ratio=float(tcfg.get("replay_mix_ratio", 0.25)),
    )


def _build_ppo(cfg: dict[str, Any], output_dir: str | None) -> PPOTrainerConfig:
    tcfg = cfg.get("train_rl", {}) or {}
    out = _optional_path(output_dir if output_dir else tcfg.get("output_dir"))
    return PPOTrainerConfig(
        n_iters=int(tcfg.get("n_iters", 30)),
        group_size=int(tcfg.get("group_size", 4)),
        prompts_per_iter=int(tcfg.get("prompts_per_iter", 2)),
        lr=float(tcfg.get("lr", 5e-3)),
        max_new_tokens=int(tcfg.get("max_new_tokens", 8)),
        temperature=float(tcfg.get("temperature", 1.0)),
        clip_eps=float(tcfg.get("clip_eps", 0.2)),
        vf_coef=float(tcfg.get("vf_coef", 0.5)),
        vf_clip_eps=float(tcfg.get("vf_clip_eps", 0.2)),
        kl_coef=float(tcfg.get("kl_coef", 0.0)),
        entropy_coef=float(tcfg.get("entropy_coef", 0.0)),
        gamma=float(tcfg.get("gamma", 1.0)),
        lam=float(tcfg.get("lam", 0.95)),
        normalize_advantage=bool(tcfg.get("normalize_advantage", True)),
        use_reference=bool(tcfg.get("use_reference", False)),
        log_every=int(tcfg.get("log_every", 5)),
        log_format=str(tcfg.get("log_format", "text")),
        save_every=int(tcfg.get("save_every", 0)),
        output_dir=out,
        seed=tcfg.get("seed", 0),
        multi_turn=bool(tcfg.get("multi_turn", False)),
        multi_turn_credit=_optional_dict(tcfg.get("multi_turn_credit")),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        profile=bool(tcfg.get("profile", False)),
        profile_output_path=_profile_output_path(tcfg, out),
        update_epochs=int(tcfg.get("update_epochs", 1)),
        minibatch_size=int(tcfg.get("minibatch_size", 0)),
        shuffle_minibatches=bool(tcfg.get("shuffle_minibatches", True)),
        loss_agg=tcfg.get("loss_agg", "mean_token"),
        max_len_for_dr_grpo=int(tcfg.get("max_len_for_dr_grpo", 256)),
        batch_generate=bool(tcfg.get("batch_generate", False)),
        interleave_sft_every=int(tcfg.get("interleave_sft_every", 0)),
        interleave_sft_samples=int(tcfg.get("interleave_sft_samples", 32)),
        interleave_sft_lr=float(tcfg.get("interleave_sft_lr", 1e-4)),
        interleave_sft_epochs=int(tcfg.get("interleave_sft_epochs", 1)),
        interleave_sft_batch_size=int(tcfg.get("interleave_sft_batch_size", 8)),
        bootstrap_sft_rounds=int(tcfg.get("bootstrap_sft_rounds", 0)),
        bootstrap_sft_samples=int(tcfg.get("bootstrap_sft_samples", 32)),
        bootstrap_sft_lr=float(tcfg.get("bootstrap_sft_lr", 1e-4)),
        bootstrap_sft_epochs=int(tcfg.get("bootstrap_sft_epochs", 1)),
        # v0.6: checkpoint / resume
        checkpoint_every=int(tcfg.get("checkpoint_every", 0)),
        keep_last_checkpoints=int(tcfg.get("keep_last_checkpoints", 3)),
        resume_from=tcfg.get("resume_from"),
        auto_resume=bool(tcfg.get("auto_resume", False)),
        # v0.6: KL estimator
        kl_estimator=_kl_estimator(tcfg.get("kl_estimator", "k1")),
        # v0.7: advantage whitening (PPO)
        whiten_advantage=bool(tcfg.get("whiten_advantage", False)),
        advantage_clip=float(tcfg.get("advantage_clip", 3.0)),
        # v0.7: best checkpoint + early stop
        save_best_checkpoint=bool(tcfg.get("save_best_checkpoint", False)),
        early_stop_patience=int(tcfg.get("early_stop_patience", 0)),
        early_stop_min_delta=float(tcfg.get("early_stop_min_delta", 1e-4)),
        target_kl=float(tcfg.get("target_kl", 0.0)),
        adaptive_kl=bool(tcfg.get("adaptive_kl", False)),
        adaptive_kl_horizon=float(tcfg.get("adaptive_kl_horizon", 10000.0)),
        adaptive_kl_min=float(tcfg.get("adaptive_kl_min", 1e-4)),
        adaptive_kl_max=float(tcfg.get("adaptive_kl_max", 10.0)),
        normalize_reward=bool(tcfg.get("normalize_reward", False)),
        reward_norm_clip=float(tcfg.get("reward_norm_clip", 10.0)),
        amp_dtype=str(tcfg.get("amp_dtype", "fp32")),
        grad_accum_steps=int(tcfg.get("grad_accum_steps", 1)),
        vllm_rollout_model=tcfg.get("vllm_rollout_model") or None,
        vllm_tensor_parallel_size=int(tcfg.get("vllm_tensor_parallel_size", 1)),
        vllm_max_model_len=int(tcfg.get("vllm_max_model_len", 4096)),
        vllm_gpu_memory_utilization=float(tcfg.get("vllm_gpu_memory_utilization", 0.90)),
        vllm_enable_prefix_caching=bool(tcfg.get("vllm_enable_prefix_caching", True)),
        vllm_sync_every=int(tcfg.get("vllm_sync_every", 1)),
        distributed_strategy=str(tcfg.get("distributed_strategy", "none")),
        fsdp_cpu_offload=bool(tcfg.get("fsdp_cpu_offload", False)),
        flash_attention=bool(tcfg.get("flash_attention", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        replay_buffer=_optional_dict(tcfg.get("replay_buffer") or cfg.get("replay_buffer")),
        replay_mix_ratio=float(tcfg.get("replay_mix_ratio", 0.25)),
    )


def _optional_axis_weights(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    return {str(k): float(v) for k, v in value.items()}


def _build_hybrid(cfg: dict[str, Any], output_dir: str | None) -> HybridTrainerConfig:
    """Build a HybridTrainerConfig from the `train_rl:` + `opd:` YAML blocks.

    The OPD branch only fires when ``opd.teacher_fill`` is true (default) AND a
    reward component emits ``opd_hint`` (e.g. ``letter_counting_next_state``).
    """
    tcfg = cfg.get("train_rl", {}) or {}
    opd_cfg = cfg.get("opd", {}) or {}
    hybrid_cfg = cfg.get("hybrid", {}) or {}
    out = _optional_path(output_dir if output_dir else tcfg.get("output_dir"))
    return HybridTrainerConfig(
        n_iters=int(tcfg.get("n_iters", 30)),
        group_size=int(tcfg.get("group_size", 4)),
        prompts_per_iter=int(tcfg.get("prompts_per_iter", 2)),
        lr=float(tcfg.get("lr", 5e-3)),
        max_new_tokens=int(tcfg.get("max_new_tokens", 8)),
        temperature=float(tcfg.get("temperature", 1.0)),
        use_reference=bool(tcfg.get("use_reference", False)),
        log_every=int(tcfg.get("log_every", 5)),
        log_format=str(tcfg.get("log_format", "text")),
        save_every=int(tcfg.get("save_every", 0)),
        output_dir=out,
        seed=tcfg.get("seed", 0),
        multi_turn=bool(tcfg.get("multi_turn", False)),
        multi_turn_credit=_optional_dict(tcfg.get("multi_turn_credit")),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        profile=bool(tcfg.get("profile", False)),
        profile_output_path=_profile_output_path(tcfg, out),
        update_epochs=int(tcfg.get("update_epochs", 1)),
        minibatch_size=int(tcfg.get("minibatch_size", 0)),
        shuffle_minibatches=bool(tcfg.get("shuffle_minibatches", True)),
        # --- hybrid weights ---
        w_rl=float(tcfg.get("w_rl", 1.0)),
        w_opd=float(tcfg.get("w_opd", 1.0)),
        w_rl_schedule=_optional_dict(tcfg.get("w_rl_schedule") or hybrid_cfg.get("w_rl_schedule")),
        w_opd_schedule=_optional_dict(
            tcfg.get("w_opd_schedule") or hybrid_cfg.get("w_opd_schedule")
        ),
        # --- GRPO branch ---
        clip_eps=float(tcfg.get("clip_eps", 0.2)),
        clip_eps_high=float(tcfg.get("clip_eps_high", 0.28)),
        kl_coef=float(tcfg.get("kl_coef", 0.02)),
        entropy_coef=float(tcfg.get("entropy_coef", 0.0)),
        advantage_norm=tcfg.get("advantage_norm", "group"),
        loss_agg=tcfg.get("loss_agg", "mean_token"),
        per_token_advantage=bool(tcfg.get("per_token_advantage", False)),
        kl_estimator=_kl_estimator(tcfg.get("kl_estimator", "k3")),
        # --- OPD branch ---
        opd_kl_coef=float(opd_cfg.get("kl_coef", 0.02)),
        opd_clip_eps=float(opd_cfg.get("clip_eps", 0.2)),
        opd_clip_eps_high=float(opd_cfg.get("clip_eps_high", 0.28)),
        opd_adv_diff_clip=float(opd_cfg.get("adv_diff_clip", 1.0)),
        opd_skip_missing_hints=bool(opd_cfg.get("skip_missing_hints", False)),
        # --- OPD teacher-logprob closed loop ---
        opd_teacher_fill=bool(opd_cfg.get("teacher_fill", True)),
        opd_hint_template=str(opd_cfg.get("hint_template", "\n\n[HINT_START]{hint}[HINT_END]\n")),
        opd_teacher_max_hint_tokens=int(opd_cfg.get("max_hint_tokens", 128)),
        opd_capability_axis_weights=_optional_axis_weights(opd_cfg.get("capability_axis_weights")),
        opd_hint_extractor=_optional_dict(opd_cfg.get("hint_extractor")),
        # --- SFT / checkpoint / shared controls ---
        interleave_sft_every=int(tcfg.get("interleave_sft_every", 0)),
        interleave_sft_samples=int(tcfg.get("interleave_sft_samples", 32)),
        interleave_sft_lr=float(tcfg.get("interleave_sft_lr", 1e-4)),
        interleave_sft_epochs=int(tcfg.get("interleave_sft_epochs", 1)),
        interleave_sft_batch_size=int(tcfg.get("interleave_sft_batch_size", 8)),
        bootstrap_sft_rounds=int(tcfg.get("bootstrap_sft_rounds", 0)),
        batch_generate=bool(tcfg.get("batch_generate", False)),
        checkpoint_every=int(tcfg.get("checkpoint_every", 0)),
        keep_last_checkpoints=int(tcfg.get("keep_last_checkpoints", 3)),
        resume_from=tcfg.get("resume_from"),
        auto_resume=bool(tcfg.get("auto_resume", False)),
        save_best_checkpoint=bool(tcfg.get("save_best_checkpoint", False)),
        early_stop_patience=int(tcfg.get("early_stop_patience", 0)),
        early_stop_min_delta=float(tcfg.get("early_stop_min_delta", 1e-4)),
        target_kl=float(tcfg.get("target_kl", 0.0)),
        adaptive_kl=bool(tcfg.get("adaptive_kl", False)),
        normalize_reward=bool(tcfg.get("normalize_reward", False)),
        reward_norm_clip=float(tcfg.get("reward_norm_clip", 10.0)),
        stability_preset=str(tcfg.get("stability_preset", "none")),
        amp_dtype=str(tcfg.get("amp_dtype", "fp32")),
        grad_accum_steps=int(tcfg.get("grad_accum_steps", 1)),
        vllm_rollout_model=tcfg.get("vllm_rollout_model") or None,
        vllm_sync_every=int(tcfg.get("vllm_sync_every", 1)),
        distributed_strategy=str(tcfg.get("distributed_strategy", "none")),
        flash_attention=bool(tcfg.get("flash_attention", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        replay_buffer=_optional_dict(tcfg.get("replay_buffer") or cfg.get("replay_buffer")),
        replay_mix_ratio=float(tcfg.get("replay_mix_ratio", 0.25)),
    )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def run_train_rl(config_path: str, output_dir: str | None = None) -> int:
    cfg = _load_yaml(config_path)
    algo = str(cfg.get("algo") or "grpo").lower()
    backend_name = _backend_name(cfg)
    env_cfg = cfg.get("environment", {}) or {}
    env_type = str(env_cfg.get("type", "echo"))
    env, reward_manager = _build_env_and_rewards(cfg)
    agent_loop_factory = _make_agent_loop_factory(cfg)

    dashboard_cfg = cfg.get("dashboard", {}) or {}
    dashboard: LiveDashboard | None = None
    extra_sinks: list[Any] = []
    if bool(dashboard_cfg.get("enabled", False)):
        dashboard = LiveDashboard(
            host=str(dashboard_cfg.get("host", "127.0.0.1")),
            port=int(dashboard_cfg.get("port", 8765)),
        )
        url = dashboard.start()
        print(f"[train-rl] dashboard live at {url}")
        extra_sinks.append(dashboard.record)

    # Resolve metrics output dir (for jsonl/tb defaults): CLI arg > tcfg > None.
    _tcfg_block = cfg.get("train_rl", {}) or {}
    _metrics_base = output_dir or _tcfg_block.get("output_dir")
    default_run_name = Path(str(_metrics_base)).name if _metrics_base else Path(config_path).stem
    metrics_writer = build_writer_from_config(
        cfg.get("metrics"),
        output_dir=_metrics_base,
        extra=extra_sinks or None,
        wandb_context={
            "name": default_run_name,
            "job_type": "train-rl",
            "tags": [algo, backend_name, env_type],
            "command": "train-rl",
            "config_path": str(config_path),
            "output_dir": str(_metrics_base) if _metrics_base else None,
            "config": cfg,
        },
    )
    metrics_sink = metrics_writer  # may be None if no dashboard + no metrics cfg
    if metrics_sink is None and extra_sinks:
        # Preserve legacy behavior: if only dashboard is configured, keep a
        # plain callable sink for backward compat.
        metrics_sink = extra_sinks[0]

    backend: LLMBackend
    trainer: GRPOTrainer | PPOTrainer | HybridTrainer
    trainer_cfg: GRPOTrainerConfig | PPOTrainerConfig | HybridTrainerConfig
    if algo == "hybrid":
        backend = _build_backend(cfg, need_value_head=False)
        hybrid_cfg = _build_hybrid(cfg, output_dir)
        hybrid_cfg.metrics_sink = metrics_sink
        trainer_cfg = hybrid_cfg
        print(
            f"[train-rl] algo=hybrid backend={backend_name} device={backend.device} "
            f"params={backend.num_parameters()} "
            f"iters={hybrid_cfg.n_iters} group={hybrid_cfg.group_size} "
            f"w_rl={hybrid_cfg.w_rl} w_opd={hybrid_cfg.w_opd} "
            f"teacher_fill={hybrid_cfg.opd_teacher_fill}"
        )
        trainer = HybridTrainer(
            policy=backend,
            env=env,
            reward_manager=reward_manager,
            cfg=hybrid_cfg,
            agent_loop_factory=agent_loop_factory,
        )
    elif algo == "grpo":
        backend = _build_backend(cfg, need_value_head=False)
        grpo_cfg = _build_grpo(cfg, output_dir)
        grpo_cfg.metrics_sink = metrics_sink
        trainer_cfg = grpo_cfg
        print(
            f"[train-rl] algo=grpo backend={backend_name} device={backend.device} "
            f"params={backend.num_parameters()} "
            f"iters={grpo_cfg.n_iters} group={grpo_cfg.group_size} lr={grpo_cfg.lr}"
        )
        trainer = GRPOTrainer(
            policy=backend,
            env=env,
            reward_manager=reward_manager,
            cfg=grpo_cfg,
            agent_loop_factory=agent_loop_factory,
        )
    elif algo == "ppo":
        backend = _build_backend(cfg, need_value_head=True)
        ppo_cfg = _build_ppo(cfg, output_dir)
        ppo_cfg.metrics_sink = metrics_sink
        trainer_cfg = ppo_cfg
        print(
            f"[train-rl] algo=ppo backend={backend_name} device={backend.device} "
            f"params={backend.num_parameters()} "
            f"iters={ppo_cfg.n_iters} group={ppo_cfg.group_size} lr={ppo_cfg.lr} "
            f"vf_coef={ppo_cfg.vf_coef} lam={ppo_cfg.lam}"
        )
        trainer = PPOTrainer(
            policy=backend,
            env=env,
            reward_manager=reward_manager,
            cfg=ppo_cfg,
            agent_loop_factory=agent_loop_factory,
        )
    else:
        raise RuntimeError(f"unknown algo: {algo!r} (expected 'grpo' or 'ppo')")

    try:
        stats = trainer.train()
    finally:
        if dashboard is not None:
            # keep dashboard alive briefly so user can inspect; comment out
            # next line to keep serving forever.
            dashboard.stop()
    try:
        if trainer_cfg.output_dir is not None:
            trainer_cfg.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = trainer_cfg.output_dir / "train_rl_summary.json"
            summary_payload = {
                "algo": algo,
                "backend": backend_name,
                "device": str(backend.device),
                "iters": stats.iters,
                "last_mean_reward": stats.last_reward(),
                "best_mean_reward": stats.best_reward(),
                "reward_delta": stats.mean_reward_delta(),
                "curriculum_snapshot": (env.snapshot() if hasattr(env, "snapshot") else None),
            }
            summary_path.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if isinstance(metrics_writer, MultiMetricsWriter):
                metrics_writer.update_summary(summary_payload)
            print(f"[train-rl] summary saved to {summary_path}")

        print(
            f"[train-rl] DONE algo={algo} last_mean_reward={stats.last_reward():.4f} "
            f"best={stats.best_reward():.4f} delta={stats.mean_reward_delta():+.4f}"
        )
        return 0
    finally:
        if isinstance(metrics_writer, MultiMetricsWriter):
            metrics_writer.close()
