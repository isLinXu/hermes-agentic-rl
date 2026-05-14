from __future__ import annotations

import json
import sys
from pathlib import Path

from hermes_agentic_rl.cli.main import main
from hermes_agentic_rl.eval import benchmark_suite


def _write_eval_summary(
    output_dir: str | None,
    *,
    score: float,
    recommendation: str = "promote",
) -> None:
    assert output_dir is not None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "eval-rl",
        "rank_metric": "mean_reward",
        "best_policy": {
            "name": "candidate",
            "score": score,
            "metrics": {
                "mean_reward": score,
                "success_rate": score,
            },
        },
        "promotion_readout": {
            "recommendation": recommendation,
            "passed": recommendation == "promote",
            "baseline": "baseline",
            "candidate": "candidate",
            "reward_delta": score - 0.5,
            "success_rate_delta": score - 0.5,
            "rank_metric_delta": score - 0.5,
            "capability_deltas": {
                "task_success": score - 0.5,
            },
        },
        "policies": [],
        "comparisons": [],
    }
    (out_dir / "eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_benchmark_suite_cli_writes_scorecard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool_config = tmp_path / "tool_eval.yaml"
    context_config = tmp_path / "context_eval.yaml"
    tool_config.write_text("eval_rl: {}\n", encoding="utf-8")
    context_config.write_text("eval_rl: {}\n", encoding="utf-8")

    def fake_eval_runner(config_path: str, output_dir: str | None = None) -> int:
        score = 0.82 if "tool" in config_path else 0.42
        recommendation = "promote" if "tool" in config_path else "hold"
        _write_eval_summary(output_dir, score=score, recommendation=recommendation)
        return 0

    monkeypatch.setattr(benchmark_suite, "run_eval_rl", fake_eval_runner)

    suite_output = tmp_path / "suite"
    config_path = tmp_path / "benchmark_suite.yaml"
    config_path.write_text(
        f"""
benchmark_suite:
  name: agentic-regression
  output_dir: {suite_output}
  benchmarks:
    - name: tool-call-validity
      config_path: {tool_config}
      required: true
      require_promotion: true
      thresholds:
        min_score: 0.8
        min_success_rate: 0.8
    - name: context-retention
      config_path: {context_config}
      required: false
      thresholds:
        min_score: 0.9
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-agentic-rl", "benchmark-suite", "--config", str(config_path)],
    )

    assert main() == 0
    scorecard = json.loads((suite_output / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["suite_name"] == "agentic-regression"
    assert scorecard["passed"] is True
    assert scorecard["benchmarks_total"] == 2
    assert scorecard["required_passed"] == 1
    assert scorecard["status_counts"] == {"failed": 1, "passed": 1}
    assert scorecard["benchmarks"][0]["promotion"]["recommendation"] == "promote"
    assert scorecard["benchmarks"][1]["failed_checks"] == ["min_score"]
    assert "| tool-call-validity | passed |" in (
        suite_output / "scorecard.md"
    ).read_text(encoding="utf-8")


def test_benchmark_suite_cli_fails_on_required_threshold_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text("eval_rl: {}\n", encoding="utf-8")

    def fake_eval_runner(_config_path: str, output_dir: str | None = None) -> int:
        _write_eval_summary(output_dir, score=0.3, recommendation="hold")
        return 0

    monkeypatch.setattr(benchmark_suite, "run_eval_rl", fake_eval_runner)

    suite_output = tmp_path / "suite"
    config_path = tmp_path / "benchmark_suite.yaml"
    config_path.write_text(
        f"""
benchmark_suite:
  output_dir: {suite_output}
  benchmarks:
    - name: required-benchmark
      config_path: {eval_config}
      required: true
      require_promotion: true
      thresholds:
        min_score: 0.8
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-agentic-rl", "benchmark-suite", "--config", str(config_path)],
    )

    assert main() == 3
    scorecard = json.loads((suite_output / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["passed"] is False
    assert scorecard["benchmarks"][0]["failed_checks"] == [
        "min_score",
        "require_promotion",
    ]


def test_benchmark_suite_cli_records_eval_runner_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text("eval_rl: {}\n", encoding="utf-8")

    def fake_eval_runner(_config_path: str, _output_dir: str | None = None) -> int:
        raise RuntimeError("checkpoint path does not exist")

    monkeypatch.setattr(benchmark_suite, "run_eval_rl", fake_eval_runner)

    suite_output = tmp_path / "suite"
    config_path = tmp_path / "benchmark_suite.yaml"
    config_path.write_text(
        f"""
benchmark_suite:
  output_dir: {suite_output}
  benchmarks:
    - name: broken-checkpoint
      config_path: {eval_config}
      required: true
      thresholds:
        min_score: 0.8
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-agentic-rl", "benchmark-suite", "--config", str(config_path)],
    )

    assert main() == 3
    scorecard = json.loads((suite_output / "scorecard.json").read_text(encoding="utf-8"))
    benchmark = scorecard["benchmarks"][0]
    assert benchmark["status"] == "failed"
    assert benchmark["failed_checks"] == ["eval_completed", "min_score"]
    assert "checkpoint path does not exist" in benchmark["error"]


def test_benchmark_suite_resolves_repo_relative_child_configs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    eval_config = configs_dir / "eval.yaml"
    eval_config.write_text("eval_rl: {}\n", encoding="utf-8")

    suite_output = tmp_path / "suite"
    suite_config = configs_dir / "benchmark_suite.yaml"
    suite_config.write_text(
        f"""
benchmark_suite:
  output_dir: {suite_output}
  benchmarks:
    - name: repo-relative-child-config
      config_path: configs/eval.yaml
      required: true
      thresholds:
        min_score: 0.8
""",
        encoding="utf-8",
    )

    seen_config_paths: list[str] = []

    def fake_eval_runner(config_path: str, output_dir: str | None = None) -> int:
        seen_config_paths.append(config_path)
        _write_eval_summary(output_dir, score=0.9)
        return 0

    monkeypatch.chdir(tmp_path)

    assert (
        benchmark_suite.run_benchmark_suite(
            str(suite_config),
            eval_runner=fake_eval_runner,
        )
        == 0
    )
    assert seen_config_paths == [str(eval_config.resolve())]
