from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.cli.main import main
from hermes_agentic_rl.eval.rl_eval import (
    _promotion_readout,
    _success_metric_score,
    select_items_by_group,
)


def _trace_row(idx: int) -> dict:
    prefix = (
        "<think>\n"
        "</think>\n"
        "<tool_call>\n"
        '{"name": "terminal", "arguments": {"command": "'
    )
    return {
        "task_id": f"trace-{idx}",
        "category": "terminal",
        "subcategory": "shell",
        "task": f"Run command {idx}",
        "tools": [{"name": "terminal", "description": "Run a shell command"}],
        "conversations": [
            {"from": "system", "value": "You are Hermes."},
            {"from": "human", "value": f"Run command {idx}."},
            {"from": "gpt", "value": f'{prefix}echo {idx}"}}}}\n</tool_call>'},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_select_items_by_group_keeps_source_trace_together() -> None:
    items = [
        {"task_id": "a-0", "source_trace_id": "a"},
        {"task_id": "a-1", "source_trace_id": "a"},
        {"task_id": "b-0", "source_trace_id": "b"},
        {"task_id": "b-1", "source_trace_id": "b"},
        {"task_id": "c-0", "source_trace_id": "c"},
        {"task_id": "c-1", "source_trace_id": "c"},
        {"task_id": "d-0", "source_trace_id": "d"},
        {"task_id": "d-1", "source_trace_id": "d"},
        {"task_id": "e-0", "source_trace_id": "e"},
        {"task_id": "e-1", "source_trace_id": "e"},
        {"task_id": "f-0", "source_trace_id": "f"},
        {"task_id": "f-1", "source_trace_id": "f"},
    ]

    selected, info = select_items_by_group(
        items,
        split="val",
        group_key="source_trace_id",
        val_ratio=1 / 3,
        test_ratio=1 / 6,
        seed=123,
    )

    selected_groups = {item["source_trace_id"] for item in selected}
    assert info["selected_groups"] == 2
    for group in selected_groups:
        assert [item["source_trace_id"] for item in selected].count(group) == 2


def test_success_metric_score_can_use_structured_metadata() -> None:
    score, found = _success_metric_score(
        reward=0.1,
        component_scores={"hermes_reasoning_trace_match": 0.1},
        component_metadata={
            "hermes_reasoning_trace_match": {
                "argument_value_similarity": 0.42,
            },
        },
        metric="metadata/argument_value_similarity",
    )

    assert found is True
    assert score == 0.42


def test_promotion_readout_can_gate_on_capability_axis_delta() -> None:
    readout = _promotion_readout(
        policy_reports=[
            {
                "name": "baseline",
                "metrics": {"mean_reward": 0.50, "success_rate": 0.50},
            },
            {
                "name": "candidate",
                "metrics": {"mean_reward": 0.55, "success_rate": 0.55},
            },
        ],
        comparisons=[
            {
                "baseline": "baseline",
                "candidate": "candidate",
                "metric_delta": {"mean_reward": 0.05, "success_rate": 0.05},
                "reward_ab": {"winner": "candidate"},
            }
        ],
        rank_metric="mean_reward",
        eval_cfg={
            "promotion_gate": {
                "required_capability_axes": ["tool_use_reliability"],
                "min_capability_delta": 0.02,
                "max_capability_regression": 0.01,
            }
        },
        capability_report={
            "deltas": {
                "tool_use_reliability": {"baseline": 0.4, "candidate": 0.43, "delta": 0.03},
                "task_success": {"baseline": 0.5, "candidate": 0.5, "delta": 0.0},
            }
        },
    )

    assert readout is not None
    assert readout["recommendation"] == "promote"
    assert readout["capability_deltas"]["tool_use_reliability"] == 0.03
    assert readout["checks"]["capability_axis/tool_use_reliability/min_delta"]["passed"] is True
    assert readout["checks"]["capability_axis/task_success/max_regression"]["passed"] is True


def test_promotion_readout_holds_when_required_capability_axis_regresses() -> None:
    readout = _promotion_readout(
        policy_reports=[
            {
                "name": "baseline",
                "metrics": {"mean_reward": 0.50, "success_rate": 0.50},
            },
            {
                "name": "candidate",
                "metrics": {"mean_reward": 0.55, "success_rate": 0.55},
            },
        ],
        comparisons=[
            {
                "baseline": "baseline",
                "candidate": "candidate",
                "metric_delta": {"mean_reward": 0.05, "success_rate": 0.05},
                "reward_ab": {"winner": "candidate"},
            }
        ],
        rank_metric="mean_reward",
        eval_cfg={
            "promotion_gate": {
                "required_capability_axes": ["tool_use_reliability"],
                "min_capability_delta": 0.01,
                "max_capability_regression": 0.02,
            }
        },
        capability_report={
            "deltas": {
                "tool_use_reliability": {"baseline": 0.4, "candidate": 0.38, "delta": -0.02},
                "task_success": {"baseline": 0.5, "candidate": 0.47, "delta": -0.03},
            }
        },
    )

    assert readout is not None
    assert readout["recommendation"] == "hold"
    assert "capability_axis/tool_use_reliability/min_delta" in readout["failed_checks"]
    assert "capability_axis/task_success/max_regression" in readout["failed_checks"]


def test_eval_rl_cli_writes_summary_rollouts_and_leaderboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "traces.jsonl"
    _write_jsonl(dataset_path, [_trace_row(i) for i in range(6)])

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_dir / "iter_00001" / "model.pt"
    checkpoint_path_2 = checkpoint_dir / "iter_00002" / "model.pt"
    checkpoint_path_3 = checkpoint_dir / "policy_iter_0003.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path_2.parent.mkdir(parents=True)
    checkpoint_path_3.parent.mkdir(parents=True, exist_ok=True)
    backend = TinyCausalLMBackend(
        TinyBackendConfig(seed=99, dim=16, n_heads=2, n_layers=1, max_len=512)
    )
    torch.save(backend.model.state_dict(), checkpoint_path)
    torch.save(backend.model.state_dict(), checkpoint_path_2)
    torch.save(backend.model.state_dict(), checkpoint_path_3)

    output_dir = tmp_path / "eval_out"
    config_path = tmp_path / "eval_rl.yaml"
    config_path.write_text(
        f"""
environment:
  type: hermes_reasoning_traces
  dataset_path: {dataset_path}
  dataset_limit: 6
  shuffle: false
  history_window_messages: 4
  max_prompt_chars: 512
  require_target_substring: '{{"name": "terminal"'
  tool_call_format_hint: true
  assistant_response_prefix: |-
    <think>
    </think>
    <tool_call>
    {{"name": "terminal", "arguments": {{"command": "
  reward_mode: hybrid
  tool_call_reward_weight: 0.85
  text_reward_weight: 0.15
agent_loop:
  type: policy
  stop_strings:
    - "</tool_call>"
backend:
  name: tiny
  dim: 16
  n_heads: 2
  n_layers: 1
  max_len: 512
  device: cpu
  seed: 0
eval_rl:
  output_dir: {output_dir}
  split: val
  split_by: source_trace_id
  val_ratio: 0.34
  test_ratio: 0.16
  seed: 0
  n_rollouts: 2
  temperature: 0.0
  max_new_tokens: 4
  success_metric: metadata/argument_value_similarity
  success_threshold: 0.0
  promotion_gate:
    min_reward_delta: 10.0
    min_success_rate_delta: 1.1
    min_rank_metric_delta: 10.0
    require_paired_winner: true
  include_rollout_text: true
  policies:
    - name: baseline
    - name: candidate
      checkpoint_dir: {checkpoint_dir}
metrics:
  jsonl: true
  wandb:
    enabled: false
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-agentic-rl", "eval-rl", "--config", str(config_path)],
    )

    assert main() == 0
    summary = json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert [policy["name"] for policy in summary["policies"]] == [
        "baseline",
        "candidate:iter_00001",
        "candidate:iter_00002",
        "candidate:policy_iter_0003",
    ]
    assert summary["split"]["selected_items_after_limit"] == 2
    assert summary["rank_metric"] == "mean_reward"
    assert summary["success_metric"] == "metadata/argument_value_similarity"
    assert summary["ranking"][0]["rank"] == 1
    assert summary["comparisons"][0]["baseline"] == "baseline"
    assert summary["best_policy"]["name"] in {
        "baseline",
        "candidate:iter_00001",
        "candidate:iter_00002",
        "candidate:policy_iter_0003",
    }
    assert summary["best_policy"]["score"] == summary["ranking"][0]["score"]
    assert summary["promotion_readout"]["baseline"] == "baseline"
    assert summary["promotion_readout"]["candidate"] in {
        "candidate:iter_00001",
        "candidate:iter_00002",
        "candidate:policy_iter_0003",
    }
    assert summary["promotion_readout"]["recommendation"] == "hold"
    assert summary["promotion_readout"]["passed"] is False
    assert "min_reward_delta" in summary["promotion_readout"]["failed_checks"]
    assert "min_rank_metric_delta" in summary["promotion_readout"]["failed_checks"]
    assert summary["capability_report"]["baseline"] == "baseline"
    assert summary["capability_report"]["axes"][0]["name"] == "task_success"
    assert "tool_use_reliability" in {
        axis["name"] for axis in summary["capability_report"]["axes"]
    }
    assert "metadata/tool_call_parse_ok" in summary["policies"][0]["metrics"]
    assert "success_score_mean" in summary["policies"][0]["metrics"]
    assert (output_dir / "eval_rollouts.jsonl").exists()
    assert "| baseline |" in (output_dir / "leaderboard.md").read_text(encoding="utf-8")
    assert "| rank | name | mean_reward |" in (output_dir / "ranking.md").read_text(encoding="utf-8")
    assert "Recommendation: `hold`" in (output_dir / "promotion.md").read_text(encoding="utf-8")
    assert "| task_success |" in (output_dir / "capability_report.md").read_text(encoding="utf-8")
    assert (output_dir / "metrics.jsonl").exists()
    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["iter"] for record in metrics] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_eval_rl_cli_can_fail_on_hold_promotion_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "traces.jsonl"
    _write_jsonl(dataset_path, [_trace_row(i) for i in range(4)])

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_dir / "iter_00001" / "model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    backend = TinyCausalLMBackend(
        TinyBackendConfig(seed=99, dim=16, n_heads=2, n_layers=1, max_len=512)
    )
    torch.save(backend.model.state_dict(), checkpoint_path)

    output_dir = tmp_path / "eval_out_fail"
    config_path = tmp_path / "eval_rl_fail.yaml"
    config_path.write_text(
        f"""
environment:
  type: hermes_reasoning_traces
  dataset_path: {dataset_path}
  dataset_limit: 4
  shuffle: false
  history_window_messages: 4
  max_prompt_chars: 512
  require_target_substring: '{{"name": "terminal"'
  tool_call_format_hint: true
  assistant_response_prefix: |-
    <think>
    </think>
    <tool_call>
    {{"name": "terminal", "arguments": {{"command": "
  reward_mode: hybrid
  tool_call_reward_weight: 0.85
  text_reward_weight: 0.15
agent_loop:
  type: policy
  stop_strings:
    - "</tool_call>"
backend:
  name: tiny
  dim: 16
  n_heads: 2
  n_layers: 1
  max_len: 512
  device: cpu
  seed: 0
eval_rl:
  output_dir: {output_dir}
  split: val
  split_by: source_trace_id
  val_ratio: 0.5
  test_ratio: 0.0
  seed: 0
  n_rollouts: 2
  temperature: 0.0
  max_new_tokens: 4
  success_metric: metadata/argument_value_similarity
  success_threshold: 0.0
  promotion_gate:
    fail_on_hold: true
    min_reward_delta: 10.0
    min_success_rate_delta: 1.1
    min_rank_metric_delta: 10.0
    require_paired_winner: true
  policies:
    - name: baseline
    - name: candidate
      checkpoint_dir: {checkpoint_dir}
metrics:
  jsonl: true
  wandb:
    enabled: false
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-agentic-rl", "eval-rl", "--config", str(config_path)],
    )

    assert main() == 3
    summary = json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["promotion_readout"]["recommendation"] == "hold"
    assert summary["promotion_readout"]["gate"]["fail_on_hold"] is True


def test_eval_gate_cli_forces_fail_on_hold_and_updates_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "traces.jsonl"
    _write_jsonl(dataset_path, [_trace_row(i) for i in range(4)])

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_dir / "iter_00001" / "model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    backend = TinyCausalLMBackend(
        TinyBackendConfig(seed=99, dim=16, n_heads=2, n_layers=1, max_len=512)
    )
    torch.save(backend.model.state_dict(), checkpoint_path)

    output_dir = tmp_path / "eval_gate_out"
    config_path = tmp_path / "eval_gate.yaml"
    config_path.write_text(
        f"""
environment:
  type: hermes_reasoning_traces
  dataset_path: {dataset_path}
  dataset_limit: 4
  shuffle: false
  history_window_messages: 4
  max_prompt_chars: 512
  require_target_substring: '{{"name": "terminal"'
  tool_call_format_hint: true
  assistant_response_prefix: |-
    <think>
    </think>
    <tool_call>
    {{"name": "terminal", "arguments": {{"command": "
  reward_mode: hybrid
  tool_call_reward_weight: 0.85
  text_reward_weight: 0.15
agent_loop:
  type: policy
  stop_strings:
    - "</tool_call>"
backend:
  name: tiny
  dim: 16
  n_heads: 2
  n_layers: 1
  max_len: 512
  device: cpu
  seed: 0
eval_rl:
  output_dir: {output_dir}
  split: val
  split_by: source_trace_id
  val_ratio: 0.5
  test_ratio: 0.0
  seed: 0
  n_rollouts: 2
  temperature: 0.0
  max_new_tokens: 4
  success_metric: metadata/argument_value_similarity
  success_threshold: 0.0
  promotion_gate:
    min_reward_delta: 10.0
    min_success_rate_delta: 1.1
    min_rank_metric_delta: 10.0
    require_paired_winner: true
  policies:
    - name: baseline
    - name: candidate
      checkpoint_dir: {checkpoint_dir}
metrics:
  jsonl: true
  wandb:
    enabled: false
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-agentic-rl", "eval-gate", "--config", str(config_path)],
    )

    assert main() == 3
    summary = json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["command"] == "eval-gate"
    assert summary["promotion_readout"]["gate"]["fail_on_hold"] is True
