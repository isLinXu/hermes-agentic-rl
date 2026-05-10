"""`train-rl` CLI: real RL training on a configured env.

Supports:
  - algo: "grpo" (default) | "ppo"
  - env: "echo" | "sim_tool" | "curriculum" | "letter_counting"
  - backend: "tiny" | "hf" (HuggingFace AutoModelForCausalLM)
  - agent_loop: "policy" (default, single-turn) | "multi_turn" (tool-use)
  - optional live dashboard (pure stdlib HTTP)
  - v0.5: per-token advantage, interleaved SFT, batch generate

The existing v0.2 config schema (echo + grpo) is 100% preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from hermes_agentic_rl.agent_loop.multi_turn_loop import MultiTurnAgentLoop
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.envs.curriculum import CurriculumEnv
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.envs.letter_counting import (
    LetterCountingEnv,
    LetterCountingReward,
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
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from hermes_agentic_rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig


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
                seed=backend_cfg.get("seed", 0),
                with_value_head=need_value_head or bool(backend_cfg.get("with_value_head", False)),
            )
        )
    if name == "hf":
        return HFCausalLMBackend(
            HFBackendConfig(
                model_name_or_path=str(backend_cfg.get("model_name_or_path", "gpt2")),
                device=str(backend_cfg.get("device", "cpu")),
                dtype=str(backend_cfg.get("dtype", "float32")),
                with_value_head=need_value_head or bool(backend_cfg.get("with_value_head", False)),
                trust_remote_code=bool(backend_cfg.get("trust_remote_code", False)),
            )
        )
    raise RuntimeError(
        f"backend '{name}' not supported; choose 'tiny' or 'hf'."
    )


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
        n = int(env_cfg.get("dataset_size", 200))
        seed = int(env_cfg.get("dataset_seed", 42))
        return (
            LetterCountingEnv(max_level=env_cfg.get("max_level", 10), seed=seed, n_samples=n),
            RewardManager([LetterCountingReward(weight=1.0)]),
        )
    raise RuntimeError(f"env type '{env_type}' not supported")


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
    if env_type != "curriculum":
        env, rm_manager = _build_single_env(env_cfg)
        extra = _build_reward_model_component(cfg)
        if extra is not None:
            rm_manager.rewards.append(extra)
        return env, rm_manager

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


# ----------------------------------------------------------------------------
# Agent-loop factory
# ----------------------------------------------------------------------------


def _make_agent_loop_factory(cfg: dict[str, Any]):
    loop_cfg = cfg.get("agent_loop", {}) or {}
    kind = loop_cfg.get("type", "policy")
    max_turns = int(loop_cfg.get("max_turns", 3))
    if kind == "policy":
        return None  # trainer default
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
            )

        return _factory
    raise RuntimeError(f"agent_loop type '{kind}' not supported")


# ----------------------------------------------------------------------------
# Trainer construction
# ----------------------------------------------------------------------------


def _build_grpo(cfg: dict[str, Any], output_dir: str | None) -> GRPOTrainerConfig:
    tcfg = cfg.get("train_rl", {}) or {}
    out = Path(output_dir) if output_dir else (
        Path(tcfg.get("output_dir")) if tcfg.get("output_dir") else None
    )
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
        save_every=int(tcfg.get("save_every", 0)),
        output_dir=out,
        seed=tcfg.get("seed", 0),
        multi_turn=bool(tcfg.get("multi_turn", False)),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        loss_agg=tcfg.get("loss_agg", "mean_token"),
        max_len_for_dr_grpo=int(tcfg.get("max_len_for_dr_grpo", 256)),
        # v0.5: per-token advantage + interleaved SFT
        per_token_advantage=bool(tcfg.get("per_token_advantage", False)),
        advantage_norm=tcfg.get("advantage_norm", "group"),
        interleave_sft_every=int(tcfg.get("interleave_sft_every", 0)),
        interleave_sft_samples=int(tcfg.get("interleave_sft_samples", 32)),
        interleave_sft_lr=float(tcfg.get("interleave_sft_lr", 1e-4)),
        # v0.5: batch generate
        batch_generate=bool(tcfg.get("batch_generate", False)),
        # v0.6: checkpoint / resume
        checkpoint_every=int(tcfg.get("checkpoint_every", 0)),
        keep_last_checkpoints=int(tcfg.get("keep_last_checkpoints", 3)),
        resume_from=tcfg.get("resume_from"),
        auto_resume=bool(tcfg.get("auto_resume", False)),
        # v0.6: KL estimator
        kl_estimator=str(tcfg.get("kl_estimator", "k1")),
        # v0.7: best checkpoint + early stop
        save_best_checkpoint=bool(tcfg.get("save_best_checkpoint", False)),
        early_stop_patience=int(tcfg.get("early_stop_patience", 0)),
        early_stop_min_delta=float(tcfg.get("early_stop_min_delta", 1e-4)),
    )


def _build_ppo(cfg: dict[str, Any], output_dir: str | None) -> PPOTrainerConfig:
    tcfg = cfg.get("train_rl", {}) or {}
    out = Path(output_dir) if output_dir else (
        Path(tcfg.get("output_dir")) if tcfg.get("output_dir") else None
    )
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
        save_every=int(tcfg.get("save_every", 0)),
        output_dir=out,
        seed=tcfg.get("seed", 0),
        multi_turn=bool(tcfg.get("multi_turn", False)),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        loss_agg=tcfg.get("loss_agg", "mean_token"),
        max_len_for_dr_grpo=int(tcfg.get("max_len_for_dr_grpo", 256)),
        # v0.6: checkpoint / resume
        checkpoint_every=int(tcfg.get("checkpoint_every", 0)),
        keep_last_checkpoints=int(tcfg.get("keep_last_checkpoints", 3)),
        resume_from=tcfg.get("resume_from"),
        auto_resume=bool(tcfg.get("auto_resume", False)),
        # v0.6: KL estimator
        kl_estimator=str(tcfg.get("kl_estimator", "k1")),
        # v0.7: advantage whitening (PPO)
        whiten_advantage=bool(tcfg.get("whiten_advantage", False)),
        advantage_clip=float(tcfg.get("advantage_clip", 3.0)),
        # v0.7: best checkpoint + early stop
        save_best_checkpoint=bool(tcfg.get("save_best_checkpoint", False)),
        early_stop_patience=int(tcfg.get("early_stop_patience", 0)),
        early_stop_min_delta=float(tcfg.get("early_stop_min_delta", 1e-4)),
    )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def run_train_rl(config_path: str, output_dir: str | None = None) -> int:
    cfg = _load_yaml(config_path)
    algo = str(cfg.get("algo") or "grpo").lower()
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
    metrics_writer = build_writer_from_config(
        cfg.get("metrics"),
        output_dir=_metrics_base,
        extra=extra_sinks or None,
    )
    metrics_sink = metrics_writer  # may be None if no dashboard + no metrics cfg
    if metrics_sink is None and extra_sinks:
        # Preserve legacy behavior: if only dashboard is configured, keep a
        # plain callable sink for backward compat.
        metrics_sink = extra_sinks[0]

    if algo == "grpo":
        backend = _build_backend(cfg, need_value_head=False)
        tcfg = _build_grpo(cfg, output_dir)
        tcfg.metrics_sink = metrics_sink
        print(
            f"[train-rl] algo=grpo backend=tiny params={backend.num_parameters()} "
            f"iters={tcfg.n_iters} group={tcfg.group_size} lr={tcfg.lr}"
        )
        trainer = GRPOTrainer(
            policy=backend, env=env, reward_manager=reward_manager, cfg=tcfg,
            agent_loop_factory=agent_loop_factory,
        )
    elif algo == "ppo":
        backend = _build_backend(cfg, need_value_head=True)
        tcfg = _build_ppo(cfg, output_dir)
        tcfg.metrics_sink = metrics_sink
        print(
            f"[train-rl] algo=ppo backend=tiny params={backend.num_parameters()} "
            f"iters={tcfg.n_iters} group={tcfg.group_size} lr={tcfg.lr} "
            f"vf_coef={tcfg.vf_coef} lam={tcfg.lam}"
        )
        trainer = PPOTrainer(
            policy=backend, env=env, reward_manager=reward_manager, cfg=tcfg,
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
        if isinstance(metrics_writer, MultiMetricsWriter):
            metrics_writer.close()

    if tcfg.output_dir is not None:
        tcfg.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = tcfg.output_dir / "train_rl_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "algo": algo,
                    "iters": stats.iters,
                    "last_mean_reward": stats.last_reward(),
                    "best_mean_reward": stats.best_reward(),
                    "reward_delta": stats.mean_reward_delta(),
                    "curriculum_snapshot": (
                        env.snapshot() if hasattr(env, "snapshot") else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[train-rl] summary saved to {summary_path}")

    print(
        f"[train-rl] DONE algo={algo} last_mean_reward={stats.last_reward():.4f} "
        f"best={stats.best_reward():.4f} delta={stats.mean_reward_delta():+.4f}"
    )
    return 0
