"""一键复验真实 benchmark 的辅助脚本。"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_RELATIVE_PATH = Path("data/hermes_reasoning_traces/train.parquet")

EVAL_CONFIG_PATH = "configs/hermes_reasoning_traces_eval_rl.yaml"
STAGE2_EVAL_CONFIG_PATH = "configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml"
BENCHMARK_SUITE_CONFIG_PATH = "configs/benchmark_suite.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/reverify_real_benchmark.py",
        description="检查真实 parquet 数据并顺序重跑 preflight / eval-rl / benchmark-suite。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="可选：真实 train.parquet 路径。提供后会链接到默认数据入口。",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过 hermes-preflight。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的动作和命令，不实际执行。",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="lambda/hermes-agent-reasoning-traces",
        help="默认 parquet 缺失时用于 dry-run 展示的 Hugging Face 数据集 repo id。",
    )
    parser.add_argument(
        "--hf-config-name",
        default="kimi",
        help="默认 parquet 缺失时用于 dry-run 展示的 Hugging Face config name。",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="默认 parquet 缺失时用于 dry-run 展示的 Hugging Face split。",
    )
    return parser


def build_commands(repo_root: Path, include_preflight: bool = True) -> list[list[str]]:
    del repo_root
    base = ["uv", "run", "python", "-m", "hermes_agentic_rl.cli.main"]
    commands: list[list[str]] = []
    if include_preflight:
        commands.append([*base, "hermes-preflight"])
    commands.append([*base, "eval-rl", "--config", EVAL_CONFIG_PATH])
    commands.append([*base, "eval-rl", "--config", STAGE2_EVAL_CONFIG_PATH])
    commands.append([*base, "benchmark-suite", "--config", BENCHMARK_SUITE_CONFIG_PATH])
    return commands


def ensure_dataset_link(repo_root: Path, dataset_override: Path | None) -> Path:
    default_dataset = repo_root / DEFAULT_DATASET_RELATIVE_PATH
    if dataset_override is None:
        if default_dataset.exists():
            return default_dataset
        raise FileNotFoundError(f"缺少数据文件：{default_dataset}")

    source = dataset_override.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"指定的数据文件不存在：{source}")

    default_dataset.parent.mkdir(parents=True, exist_ok=True)
    if default_dataset.exists() or default_dataset.is_symlink():
        if default_dataset.resolve() == source:
            return default_dataset
        default_dataset.unlink()

    default_dataset.symlink_to(source)
    return default_dataset


def _resolve_data_source(
    repo_root: Path,
    dataset_override: Path | None,
    *,
    hf_repo_id: str,
    hf_config_name: str,
    hf_split: str,
) -> tuple[str, str]:
    default_dataset = repo_root / DEFAULT_DATASET_RELATIVE_PATH
    if dataset_override is not None:
        source = dataset_override.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"指定的数据文件不存在：{source}")
        return "parquet", f"显式数据文件：{source}"
    if default_dataset.exists():
        return "parquet", f"默认数据文件：{default_dataset}"
    return "hf", f"HF 数据源：{hf_repo_id}/{hf_config_name}/{hf_split}"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def run_commands(repo_root: Path, commands: list[list[str]]) -> int:
    env = _clean_env()
    for command in commands:
        print(f"[RUN] {' '.join(shlex.quote(part) for part in command)}")
        subprocess.run(command, cwd=repo_root, env=env, check=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    include_preflight = not args.skip_preflight
    commands = build_commands(PROJECT_ROOT, include_preflight=include_preflight)

    try:
        source_mode, source_description = _resolve_data_source(
            PROJECT_ROOT,
            args.dataset,
            hf_repo_id=args.hf_repo_id,
            hf_config_name=args.hf_config_name,
            hf_split=args.hf_split,
        )
        if args.dry_run:
            print(f"数据源模式: {source_mode}")
            print(f"数据源说明: {source_description}")
            for command in commands:
                print("  " + " ".join(shlex.quote(part) for part in command))
            return 0

        if source_mode == "hf":
            raise FileNotFoundError(f"缺少数据文件：{PROJECT_ROOT / DEFAULT_DATASET_RELATIVE_PATH}")

        resolved_dataset = ensure_dataset_link(PROJECT_ROOT, args.dataset)
        print(f"[DATASET] 已就绪：{resolved_dataset}")
        return run_commands(PROJECT_ROOT, commands)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，退出码={exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
