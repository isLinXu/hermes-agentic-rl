from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_build_commands_uses_runtime_config_paths() -> None:
    from scripts.reverify_real_benchmark import build_commands

    commands = build_commands(
        eval_config_path=Path("/tmp/eval.yaml"),
        stage2_eval_config_path=Path("/tmp/stage2.yaml"),
        benchmark_suite_config_path=Path("/tmp/suite.yaml"),
        include_preflight=True,
    )

    assert len(commands) == 4
    assert commands[0][-1] == "hermes-preflight"
    assert commands[1][-2:] == ["--config", "/tmp/eval.yaml"]
    assert commands[2][-2:] == ["--config", "/tmp/stage2.yaml"]
    assert commands[3][-2:] == ["--config", "/tmp/suite.yaml"]


def test_ensure_dataset_link_creates_default_symlink(tmp_path: Path) -> None:
    from scripts.reverify_real_benchmark import DEFAULT_DATASET_RELATIVE_PATH, ensure_dataset_link

    repo_root = tmp_path
    source_dataset = tmp_path / "downloads" / "train.parquet"
    source_dataset.parent.mkdir(parents=True)
    source_dataset.write_bytes(b"parquet")

    resolved = ensure_dataset_link(repo_root=repo_root, dataset_override=source_dataset)

    expected_link = repo_root / DEFAULT_DATASET_RELATIVE_PATH
    assert resolved == expected_link
    assert expected_link.is_symlink()
    assert expected_link.resolve() == source_dataset.resolve()


def test_main_dry_run_prints_plan_with_dataset_override(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import scripts.reverify_real_benchmark as module

    repo_root = tmp_path
    dataset = tmp_path / "external" / "train.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"parquet")
    monkeypatch.setattr(module, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reverify_real_benchmark.py",
            "--dry-run",
            "--dataset",
            str(dataset),
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "数据源模式: parquet" in captured.out
    assert str(dataset) in captured.out
    assert "configs/hermes_reasoning_traces_eval_rl.yaml" in captured.out
    assert "configs/benchmark_suite.yaml" in captured.out


def test_main_dry_run_falls_back_to_hf_when_default_parquet_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import scripts.reverify_real_benchmark as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reverify_real_benchmark.py",
            "--dry-run",
            "--hf-repo-id",
            "lambda/hermes-agent-reasoning-traces",
            "--hf-config-name",
            "kimi",
            "--hf-split",
            "train",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "数据源模式: hf" in captured.out
    assert "lambda/hermes-agent-reasoning-traces" in captured.out


def test_explicit_dataset_keeps_parquet_priority(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import scripts.reverify_real_benchmark as module

    dataset = tmp_path / "external" / "train.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"parquet")

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reverify_real_benchmark.py",
            "--dry-run",
            "--dataset",
            str(dataset),
            "--hf-repo-id",
            "lambda/hermes-agent-reasoning-traces",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "数据源模式: parquet" in captured.out
    assert str(dataset) in captured.out


def test_materialize_runtime_configs_builds_hf_eval_configs_without_dataset_path(
    tmp_path: Path,
) -> None:
    import scripts.reverify_real_benchmark as module

    plan = module.RuntimePlan(
        mode="hf",
        description="HF fallback",
        dataset_path=None,
        dataset_override=None,
        hf_repo_id="lambda/hermes-agent-reasoning-traces",
        hf_config_name="kimi",
        hf_split="train",
        hf_streaming=True,
        hf_rows_api_only=True,
    )

    runtime = module.materialize_runtime_configs(tmp_path, plan)
    payload = module._load_yaml(runtime.eval_config_path)

    assert "dataset_path" not in payload["environment"]
    assert payload["environment"]["repo_id"] == "lambda/hermes-agent-reasoning-traces"
    assert payload["environment"]["config_name"] == "kimi"
    assert payload["environment"]["split"] == "train"
    assert payload["environment"]["streaming"] is True
    assert payload["environment"]["rows_api_only"] is True


def test_materialize_runtime_configs_rewrites_benchmark_suite_config_paths(
    tmp_path: Path,
) -> None:
    import scripts.reverify_real_benchmark as module

    plan = module.RuntimePlan(
        mode="hf",
        description="HF fallback",
        dataset_path=None,
        dataset_override=None,
        hf_repo_id="lambda/hermes-agent-reasoning-traces",
        hf_config_name="kimi",
        hf_split="train",
        hf_streaming=True,
        hf_rows_api_only=True,
    )

    runtime = module.materialize_runtime_configs(tmp_path, plan)
    payload = module._load_yaml(runtime.benchmark_suite_config_path)
    config_paths = [item["config_path"] for item in payload["benchmark_suite"]["benchmarks"]]

    assert (
        str(runtime.eval_config_path) in config_paths
        or str(runtime.stage2_eval_config_path) in config_paths
    )
    assert (
        "configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml"
        not in config_paths
    )
