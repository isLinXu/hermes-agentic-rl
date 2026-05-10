from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_agentic_rl import __version__
from hermes_agentic_rl.config import load_config
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
from hermes_agentic_rl.runtime.errors import RuntimeUnavailableError
from hermes_agentic_rl.runtime.fake_adapter import FakeRuntimeAdapter
from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter
from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-agentic-rl")
    parser.add_argument("--version", action="store_true", help="show version")
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="override dataset path (train only; overrides config.environment.dataset_path)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="override export jsonl path (train only; overrides trainer.export_training_path)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of dataset items to process (train/rollout)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for dataset shuffling (train only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite trainer export file instead of appending (train only)",
    )
    parser.add_argument(
        "--min-nonzero-ratio",
        type=float,
        default=None,
        help="minimum ratio of non-zero reward samples required to exit with 0 (train only)",
    )
    parser.add_argument(
        "--min-verifier-pass-ratio",
        type=float,
        default=None,
        help="minimum ratio of FileSystemVerifierReward pass required (train only)",
    )
    parser.add_argument(
        "--workdir-base",
        type=str,
        default=None,
        help="base directory for per-item workdirs (train only)",
    )
    parser.add_argument(
        "--workdir-clean",
        action="store_true",
        help="clean workdir-base before running (train only; restricted to ./outputs)",
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="print effective train config (after CLI overrides) then run (train only)",
    )
    parser.add_argument(
        "--print-effective-config-only",
        action="store_true",
        help="print effective train config (after CLI overrides) and exit 0 (train only)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=["help", "rollout", "train", "train-rl", "atropos-preflight", "offline"],
    )
    return parser


def _build_runtime_adapter(config: dict[str, Any]) -> Any:
    integration = config["runtime"]["integration"]
    if integration == "fake":
        return FakeRuntimeAdapter()
    if integration == "hermes":
        return HermesRuntimeAdapter()
    raise RuntimeError(f"unsupported integration: {integration}")


def _build_agent_loop(config: dict[str, Any]) -> Any:
    adapter = _build_runtime_adapter(config)
    if not adapter.is_available():
        raise RuntimeUnavailableError(adapter.describe_unavailable_reason())
    return adapter.build_agent_loop(config)


def _run_rollout(config_path: str, output_path: str | None) -> int:
    config = load_config(config_path)
    dataset = load_jsonl_dataset(config["environment"]["dataset_path"])
    item = dataset[0]
    instruction = item.get("instruction", "")
    trajectory = asyncio.run(
        RolloutManager(_build_agent_loop(config)).collect(item, instruction)
    )
    serialized = trajectory_to_dict(trajectory)

    if output_path:
        Path(output_path).write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"trajectory saved to {output_path}")
        return 0

    print(json.dumps(serialized, ensure_ascii=False))
    return 0


def _build_reward_manager(config: dict[str, Any]) -> RewardManager:
    from hermes_agentic_rl.rewards.filesystem_verifier_reward import FileSystemVerifierReward

    reward_cfg = config.get("reward", {}) if isinstance(config, dict) else {}
    components = reward_cfg.get("components")

    # 默认：更偏向 verifier（更可靠的成功信号）
    default_components = [
        {"name": "outcome_reward", "weight": 0.2},
        {"name": "toolcall_reward", "weight": 0.2},
        {"name": "filesystem_verifier_reward", "weight": 0.6},
    ]
    if not isinstance(components, list) or not components:
        components = default_components

    rewards = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        name = comp.get("name")
        weight = float(comp.get("weight", 1.0))
        if name == "outcome_reward":
            rewards.append(OutcomeReward(weight=weight))
        elif name == "toolcall_reward":
            rewards.append(ToolcallReward(weight=weight))
        elif name == "filesystem_verifier_reward":
            rewards.append(FileSystemVerifierReward(weight=weight))

    return RewardManager(rewards=rewards)


def _select_items(dataset: list[dict[str, Any]], limit: int | None, seed: int | None) -> list[dict[str, Any]]:
    items = list(dataset)
    if seed is not None:
        random.Random(seed).shuffle(items)
    if limit is None:
        return items[:1]
    if limit <= 0:
        return []
    return items[:limit]


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    prev_terminal_cwd = os.environ.get("TERMINAL_CWD")
    os.environ["TERMINAL_CWD"] = str(path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)
        if prev_terminal_cwd is None:
            os.environ.pop("TERMINAL_CWD", None)
        else:
            os.environ["TERMINAL_CWD"] = prev_terminal_cwd


def _reset_hermes_tool_caches_for_workdir() -> None:
    """Best-effort: reset hermes tool caches so file tools rebind to new workdir.

    Hermes's file tools may resolve paths against a cached "live cwd" derived from
    its terminal environment, keyed by an internal task id. When running multiple
    samples inside one Python process, that cache can cause later samples to write
    into the first sample's directory even if we chdir().

    This reset is intentionally best-effort and no-ops when hermes-agent isn't installed.
    """
    try:
        from tools import file_tools  # type: ignore

        if hasattr(file_tools, "_file_ops_cache"):
            file_tools._file_ops_cache.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from tools import terminal_tool  # type: ignore

        if hasattr(terminal_tool, "_active_environments"):
            terminal_tool._active_environments.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _clean_workdir_base(base_cwd: Path, workdir_base: Path) -> None:
    """Safely clean workdir_base contents.

    Safety constraint: only allows cleaning directories under (base_cwd / "outputs").
    Intended usage: remove stale per-task directories created by previous runs.
    """
    outputs_root = (base_cwd / "outputs").resolve()

    resolved = workdir_base
    if not resolved.is_absolute():
        resolved = base_cwd / resolved
    resolved = resolved.resolve()

    if not resolved.is_relative_to(outputs_root):
        raise RuntimeError(
            f"refusing to clean workdir_base outside outputs/: workdir_base={resolved} outputs_root={outputs_root}"
        )

    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
        return

    if not resolved.is_dir():
        raise RuntimeError(f"workdir_base is not a directory: {resolved}")

    # Only remove direct children directories (best-effort; never remove the base itself).
    for child in resolved.iterdir():
        # Never follow symlinks out of tree.
        if child.is_symlink():
            child.unlink(missing_ok=True)
            continue
        if child.is_dir():
            shutil.rmtree(child)


def _run_train(
    config_path: str,
    limit: int | None,
    seed: int | None,
    overwrite: bool,
    min_nonzero_ratio: float | None,
    min_verifier_pass_ratio: float | None,
    workdir_base: str | None,
    dataset_path: str | None,
    export_path: str | None,
    workdir_clean: bool,
    print_effective_config: bool,
    print_effective_config_only: bool,
) -> int:
    base_cwd = Path.cwd()
    config = load_config(config_path)
    reward_manager = _build_reward_manager(config)

    runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
    environment_cfg = config.get("environment", {})
    trainer_cfg = config.get("trainer", {})
    effective_limit = limit if limit is not None else int(trainer_cfg.get("max_samples", 1))
    effective_seed = seed if seed is not None else trainer_cfg.get("seed")
    effective_overwrite = overwrite or bool(trainer_cfg.get("overwrite", False))
    effective_min_nonzero_ratio = (
        float(min_nonzero_ratio)
        if min_nonzero_ratio is not None
        else float(trainer_cfg.get("min_nonzero_reward_ratio", 0.0))
    )
    effective_min_verifier_pass_ratio = (
        float(min_verifier_pass_ratio)
        if min_verifier_pass_ratio is not None
        else float(trainer_cfg.get("min_verifier_pass_ratio", 0.0))
    )
    effective_workdir_base = (
        Path(workdir_base)
        if workdir_base is not None
        else Path(trainer_cfg.get("workdir_base"))
        if trainer_cfg.get("workdir_base") is not None
        else None
    )
    if effective_workdir_base is not None and not effective_workdir_base.is_absolute():
        effective_workdir_base = (base_cwd / effective_workdir_base).resolve()
    if effective_workdir_base is not None:
        effective_workdir_base.mkdir(parents=True, exist_ok=True)

    if workdir_clean:
        if effective_workdir_base is None:
            raise RuntimeError("--workdir-clean requires workdir_base to be set (via --workdir-base or config)")
        _clean_workdir_base(base_cwd=base_cwd, workdir_base=effective_workdir_base)

    effective_dataset_path = Path(dataset_path) if dataset_path is not None else Path(environment_cfg["dataset_path"])
    if not effective_dataset_path.is_absolute():
        effective_dataset_path = (base_cwd / effective_dataset_path).resolve()

    effective_export_path = (
        Path(export_path) if export_path is not None else Path(trainer_cfg["export_training_path"])
    )
    if not effective_export_path.is_absolute():
        effective_export_path = (base_cwd / effective_export_path).resolve()

    effective_config_payload = {
        "command": "train",
        "base_cwd": str(base_cwd),
        "runtime": {
            "integration": runtime_cfg.get("integration"),
            "provider": runtime_cfg.get("provider"),
            "base_url": runtime_cfg.get("base_url"),
            "model": runtime_cfg.get("model"),
        },
        "environment": {"dataset_path": str(effective_dataset_path)},
        "trainer": {
            "export_training_path": str(effective_export_path),
            "workdir_base": str(effective_workdir_base) if effective_workdir_base is not None else None,
            "overwrite": bool(effective_overwrite),
            "max_samples": int(effective_limit) if effective_limit is not None else None,
            "seed": effective_seed,
            "min_nonzero_reward_ratio": float(effective_min_nonzero_ratio),
            "min_verifier_pass_ratio": float(effective_min_verifier_pass_ratio),
            "workdir_clean": bool(workdir_clean),
        },
    }

    if print_effective_config or print_effective_config_only:
        print(json.dumps(effective_config_payload, ensure_ascii=False))
        if print_effective_config_only:
            return 0

    dataset = load_jsonl_dataset(effective_dataset_path)
    items = _select_items(dataset, effective_limit, effective_seed)

    shared_loop = _build_agent_loop(config) if effective_workdir_base is None else None

    effective_export_path.parent.mkdir(parents=True, exist_ok=True)
    if effective_overwrite:
        effective_export_path.write_text("", encoding="utf-8")

    trainer_bridge = TrainerBridge(AtroposGrpoTrainer(output_path=effective_export_path))

    async def _run_all() -> dict[str, Any]:
        rewards: list[float] = []
        nonzero_count = 0
        verifier_total = 0
        verifier_passed = 0
        verifier_failed_samples = 0
        verifier_failures_total = 0
        verifier_failure_reasons: Counter[str] = Counter()
        for item in items:
            instruction = item.get("instruction", "")

            if effective_workdir_base is not None:
                task_id = str(item.get("task_id", "unknown"))
                item_dir = effective_workdir_base / task_id
                item_dir.mkdir(parents=True, exist_ok=True)
                with _pushd(item_dir):
                    _reset_hermes_tool_caches_for_workdir()
                    loop = _build_agent_loop(config)
                    trajectory = await RolloutManager(loop).collect(item, instruction)
                    summary = await reward_manager.evaluate(
                        item, trajectory, tool_context=None
                    )
                    await trainer_bridge.submit(item, trajectory, summary)
            else:
                trajectory = await RolloutManager(shared_loop).collect(item, instruction)
                summary = await reward_manager.evaluate(item, trajectory, tool_context=None)
                await trainer_bridge.submit(item, trajectory, summary)

            rewards.append(summary.final_score)
            if summary.final_score > 0:
                nonzero_count += 1

            # verifier 统计：仅统计提供了 expected_files 的样本
            expected_files = item.get("expected_files")
            if expected_files:
                verifier_total += 1
                for comp in summary.components:
                    if comp.name == "filesystem_verifier_reward":
                        if comp.score >= 1.0:
                            verifier_passed += 1
                        else:
                            verifier_failed_samples += 1
                            failures = comp.metadata.get("failures") if isinstance(comp.metadata, dict) else None
                            if isinstance(failures, list):
                                verifier_failures_total += len(failures)
                                for failure in failures:
                                    if isinstance(failure, dict):
                                        reason = str(failure.get("reason", "")).strip()
                                        if reason:
                                            verifier_failure_reasons[reason] += 1
                        break
        mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
        nonzero_ratio = (nonzero_count / len(rewards)) if rewards else 0.0
        verifier_pass_ratio = (
            (verifier_passed / verifier_total) if verifier_total > 0 else 0.0
        )
        return {
            "samples": len(rewards),
            "mean_reward": mean_reward,
            "nonzero_ratio": nonzero_ratio,
            "verifier_pass_ratio": verifier_pass_ratio,
            "verifier_total": verifier_total,
            "verifier_failed_samples": verifier_failed_samples,
            "verifier_failures_total": verifier_failures_total,
            "verifier_failures_per_sample": (
                (verifier_failures_total / verifier_total) if verifier_total > 0 else 0.0
            ),
            "verifier_failure_top": verifier_failure_reasons.most_common(5),
        }

    stats = asyncio.run(_run_all())

    print(f"training sample saved to {effective_export_path} (n={stats['samples']})")
    print(
        "train summary: "
        f"samples={stats['samples']} "
        f"mean_reward={stats['mean_reward']:.4f} "
        f"nonzero_ratio={stats['nonzero_ratio']:.4f}"
    )
    if stats.get("verifier_total", 0) > 0:
        print(
            "verifier summary: "
            f"verifier_total={stats['verifier_total']} "
            f"verifier_pass_ratio={stats['verifier_pass_ratio']:.4f} "
            f"verifier_failed_samples={stats.get('verifier_failed_samples', 0)} "
            f"verifier_failures_total={stats.get('verifier_failures_total', 0)} "
            f"verifier_failures_per_sample={stats.get('verifier_failures_per_sample', 0.0):.4f}"
        )
        if stats.get("verifier_failure_top"):
            formatted = ", ".join(
                f"{reason}={count}" for reason, count in stats["verifier_failure_top"]
            )
            print(f"verifier_failure_top: {formatted}")

    if stats["samples"] > 0 and stats["nonzero_ratio"] < effective_min_nonzero_ratio:
        print(
            "train quality gate failed: "
            f"nonzero_ratio {stats['nonzero_ratio']:.4f} < {effective_min_nonzero_ratio:.4f}"
        )
        return 3

    if stats.get("verifier_total", 0) > 0 and stats["verifier_pass_ratio"] < effective_min_verifier_pass_ratio:
        print(
            "train quality gate failed: "
            f"verifier_pass_ratio {stats['verifier_pass_ratio']:.4f} < {effective_min_verifier_pass_ratio:.4f}"
        )
        return 4

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print(__version__)
        return 0
    if args.command == "help":
        parser.print_help()
        return 0
    try:
        if args.command == "rollout":
            if not args.config_path:
                raise RuntimeError("--config is required for rollout")
            return _run_rollout(args.config_path, args.output_path)
        if args.command == "train":
            if not args.config_path:
                raise RuntimeError("--config is required for train")
            return _run_train(
                args.config_path,
                limit=args.limit,
                seed=args.seed,
                overwrite=bool(args.overwrite),
                min_nonzero_ratio=args.min_nonzero_ratio,
                min_verifier_pass_ratio=args.min_verifier_pass_ratio,
                workdir_base=args.workdir_base,
                dataset_path=args.dataset,
                export_path=args.export,
                workdir_clean=bool(args.workdir_clean),
                print_effective_config=bool(args.print_effective_config),
                print_effective_config_only=bool(args.print_effective_config_only),
            )
        if args.command == "train-rl":
            if not args.config_path:
                raise RuntimeError("--config is required for train-rl")
            from hermes_agentic_rl.cli.train_rl import run_train_rl

            return run_train_rl(args.config_path, output_dir=args.output_path)
        if args.command == "offline":
            if not args.config_path:
                raise RuntimeError("--config is required for offline")
            from hermes_agentic_rl.cli.offline_cli import run_offline

            return run_offline(args.config_path)
        if args.command == "atropos-preflight":
            from hermes_agentic_rl.integrations.atropos_preflight import run_atropos_preflight

            result = run_atropos_preflight(Path.cwd())
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
            return 0 if not result.missing else 1
    except RuntimeUnavailableError as exc:
        print(f"runtime unavailable: {exc}")
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    print(f"command {args.command} is wired but not yet implemented")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
