"""CLI entrypoints for offline trainers (BC, DPO, RM).

Each entrypoint takes a YAML/JSON config describing:
  - backend: TinyBackendConfig-compatible dict (with_value_head optional)
  - data_path: JSONL replay buffer
  - algo: 'bc' | 'dpo' | 'rm'
  - algo_config: fields for the trainer config
  - save_path: where to dump the resulting state_dict (torch.save)

Deliberately minimal; the typical user composes these trainers from Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _build_backend(cfg: dict[str, Any]):
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

    return TinyCausalLMBackend(TinyBackendConfig(**cfg))


def run_offline(config_path: str | Path) -> int:
    cfg = _load_config(config_path)
    algo = cfg.get("algo")
    backend = _build_backend(cfg.get("backend", {}))

    from hermes_agentic_rl.offline import (
        BCConfig,
        BCTrainer,
        DPOConfig,
        DPOTrainer,
        ReplayBuffer,
    )

    buf = ReplayBuffer.load_jsonl(cfg["data_path"])
    algo_cfg = cfg.get("algo_config", {})
    save_path = cfg.get("save_path")

    if algo == "bc":
        trainer = BCTrainer(backend, buf, cfg=BCConfig(**algo_cfg))
        trainer.train()
        if save_path:
            trainer.save_policy(save_path)
    elif algo == "dpo":
        trainer = DPOTrainer(backend, buf, cfg=DPOConfig(**algo_cfg))
        trainer.train()
        if save_path:
            trainer.save_policy(save_path)
    elif algo == "rm":
        from hermes_agentic_rl.rewards.reward_model import (
            RewardModel,
            RewardModelConfig,
            RewardModelTrainer,
        )

        rm = RewardModel(backend, freeze_base=algo_cfg.pop("freeze_base", True))
        trainer = RewardModelTrainer(rm, buf, cfg=RewardModelConfig(**algo_cfg))
        trainer.train()
        if save_path:
            trainer.save_head(save_path)
    else:
        raise ValueError(f"unknown offline algo: {algo!r}")
    print(f"[offline:{algo}] done.")
    return 0
