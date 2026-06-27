from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_build_commands_covers_preflight_eval_and_suite(tmp_path: Path) -> None:
    from scripts.reverify_real_benchmark import build_commands

    repo_root = tmp_path
    dataset_path = repo_root / "data" / "hermes_reasoning_traces" / "train.parquet"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"parquet")

    commands = build_commands(repo_root=repo_root, include_preflight=True)

    assert len(commands) == 4
    assert commands[0][-1] == "hermes-preflight"
    assert commands[1][-2:] == ["--config", "configs/hermes_reasoning_traces_eval_rl.yaml"]
    assert commands[2][-2:] == [
        "--config",
        "configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml",
    ]
    assert commands[3][-2:] == ["--config", "configs/benchmark_suite.yaml"]


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
    assert "Dry run" in captured.out
    assert "configs/hermes_reasoning_traces_eval_rl.yaml" in captured.out
    assert "configs/benchmark_suite.yaml" in captured.out


def test_main_returns_error_when_dataset_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import scripts.reverify_real_benchmark as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["reverify_real_benchmark.py", "--dry-run"])

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "train.parquet" in captured.err
