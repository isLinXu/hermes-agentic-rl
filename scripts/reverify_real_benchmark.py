"""一键复验真实 benchmark 的辅助脚本。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_RELATIVE_PATH = Path("data/hermes_reasoning_traces/train.parquet")

EVAL_CONFIG_PATH = "configs/hermes_reasoning_traces_eval_rl.yaml"
STAGE2_EVAL_CONFIG_PATH = "configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml"
BENCHMARK_SUITE_CONFIG_PATH = "configs/benchmark_suite.yaml"


@dataclass(slots=True)
class RuntimePlan:
    mode: str
    description: str
    dataset_path: Path | None
    dataset_override: Path | None
    hf_repo_id: str
    hf_config_name: str
    hf_split: str
    hf_streaming: bool
    hf_rows_api_only: bool


@dataclass(slots=True)
class RuntimeConfigs:
    eval_config_path: Path
    stage2_eval_config_path: Path
    benchmark_suite_config_path: Path
    temp_dir: Path | None = None


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/reverify_real_benchmark.py",
        description="检查真实 parquet 数据并顺序重跑 preflight / eval-rl / benchmark-suite。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="可选:真实 train.parquet 路径。提供后会链接到默认数据入口。",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过 hermes-preflight。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的动作和命令,不实际执行。",
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
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        default=True,
        help="HF 模式下启用 streaming。",
    )
    parser.add_argument(
        "--hf-rows-api-only",
        action="store_true",
        default=True,
        help="HF 模式下仅使用 rows API。",
    )
    return parser


def build_commands(
    *,
    eval_config_path: Path,
    stage2_eval_config_path: Path,
    benchmark_suite_config_path: Path,
    include_preflight: bool = True,
) -> list[list[str]]:
    base = ["uv", "run", "python", "-m", "hermes_agentic_rl.cli.main"]
    commands: list[list[str]] = []
    if include_preflight:
        commands.append([*base, "hermes-preflight"])
    commands.append([*base, "eval-rl", "--config", str(eval_config_path)])
    commands.append([*base, "eval-rl", "--config", str(stage2_eval_config_path)])
    commands.append([*base, "benchmark-suite", "--config", str(benchmark_suite_config_path)])
    return commands


def ensure_dataset_link(repo_root: Path, dataset_override: Path | None) -> Path:
    default_dataset = repo_root / DEFAULT_DATASET_RELATIVE_PATH
    if dataset_override is None:
        if default_dataset.exists():
            return default_dataset
        raise FileNotFoundError(f"缺少数据文件:{default_dataset}")

    source = dataset_override.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"指定的数据文件不存在:{source}")

    default_dataset.parent.mkdir(parents=True, exist_ok=True)
    if default_dataset.exists() or default_dataset.is_symlink():
        if default_dataset.resolve() == source:
            return default_dataset
        default_dataset.unlink()

    default_dataset.symlink_to(source)
    return default_dataset


def resolve_data_source(args: argparse.Namespace, repo_root: Path) -> RuntimePlan:
    default_dataset = repo_root / DEFAULT_DATASET_RELATIVE_PATH
    if args.dataset is not None:
        source = args.dataset.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"指定的数据文件不存在:{source}")
        return RuntimePlan(
            mode="parquet",
            description=f"显式数据文件:{source}",
            dataset_path=default_dataset,
            dataset_override=source,
            hf_repo_id=args.hf_repo_id,
            hf_config_name=args.hf_config_name,
            hf_split=args.hf_split,
            hf_streaming=args.hf_streaming,
            hf_rows_api_only=args.hf_rows_api_only,
        )
    if default_dataset.exists():
        return RuntimePlan(
            mode="parquet",
            description=f"默认数据文件:{default_dataset}",
            dataset_path=default_dataset,
            dataset_override=None,
            hf_repo_id=args.hf_repo_id,
            hf_config_name=args.hf_config_name,
            hf_split=args.hf_split,
            hf_streaming=args.hf_streaming,
            hf_rows_api_only=args.hf_rows_api_only,
        )
    return RuntimePlan(
        mode="hf",
        description=f"HF 数据源:{args.hf_repo_id}/{args.hf_config_name}/{args.hf_split}",
        dataset_path=None,
        dataset_override=None,
        hf_repo_id=args.hf_repo_id,
        hf_config_name=args.hf_config_name,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        hf_rows_api_only=args.hf_rows_api_only,
    )


def materialize_runtime_configs(repo_root: Path, plan: RuntimePlan) -> RuntimeConfigs:
    if plan.mode == "parquet":
        return RuntimeConfigs(
            eval_config_path=repo_root / EVAL_CONFIG_PATH,
            stage2_eval_config_path=repo_root / STAGE2_EVAL_CONFIG_PATH,
            benchmark_suite_config_path=repo_root / BENCHMARK_SUITE_CONFIG_PATH,
            temp_dir=None,
        )

    (repo_root / ".tmp").mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="reverify_real_benchmark_", dir=repo_root / ".tmp"))
    eval_payload = _load_yaml(TEMPLATE_ROOT / EVAL_CONFIG_PATH)
    stage2_payload = _load_yaml(TEMPLATE_ROOT / STAGE2_EVAL_CONFIG_PATH)
    suite_payload = _load_yaml(TEMPLATE_ROOT / BENCHMARK_SUITE_CONFIG_PATH)

    for payload in (eval_payload, stage2_payload):
        environment = payload["environment"]
        environment.pop("dataset_path", None)
        environment["repo_id"] = plan.hf_repo_id
        environment["config_name"] = plan.hf_config_name
        environment["split"] = plan.hf_split
        environment["streaming"] = plan.hf_streaming
        environment["rows_api_only"] = plan.hf_rows_api_only

    eval_path = temp_dir / "hermes_reasoning_traces_eval_rl.hf.yaml"
    stage2_path = temp_dir / "hermes_reasoning_traces_eval_rl_terminal_command_stage2.hf.yaml"
    _dump_yaml(eval_path, eval_payload)
    _dump_yaml(stage2_path, stage2_payload)

    for benchmark in suite_payload["benchmark_suite"]["benchmarks"]:
        config_path = benchmark["config_path"]
        if config_path == EVAL_CONFIG_PATH:
            benchmark["config_path"] = str(eval_path)
        elif config_path == STAGE2_EVAL_CONFIG_PATH:
            benchmark["config_path"] = str(stage2_path)

    suite_path = temp_dir / "benchmark_suite.hf.yaml"
    _dump_yaml(suite_path, suite_payload)
    return RuntimeConfigs(eval_path, stage2_path, suite_path, temp_dir=temp_dir)


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

    try:
        plan = resolve_data_source(args, PROJECT_ROOT)
        runtime = materialize_runtime_configs(PROJECT_ROOT, plan)
        commands = build_commands(
            eval_config_path=runtime.eval_config_path,
            stage2_eval_config_path=runtime.stage2_eval_config_path,
            benchmark_suite_config_path=runtime.benchmark_suite_config_path,
            include_preflight=include_preflight,
        )
        if args.dry_run:
            print(f"数据源模式: {plan.mode}")
            print(f"数据源说明: {plan.description}")
            if runtime.temp_dir is not None:
                print(f"临时配置目录: {runtime.temp_dir}")
            for command in commands:
                print("  " + " ".join(shlex.quote(part) for part in command))
            return 0

        if plan.mode == "parquet":
            resolved_dataset = ensure_dataset_link(PROJECT_ROOT, plan.dataset_override)
            print(f"[DATASET] 已就绪:{resolved_dataset}")
        else:
            print(f"[DATASET] 已切换到 {plan.description}")
        return run_commands(PROJECT_ROOT, commands)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败,退出码={exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
