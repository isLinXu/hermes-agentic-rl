from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend


def _trace_row(idx: int) -> dict[str, object]:
    prefix = (
        "<think>\n"
        "</think>\n"
        "<tool_call>\n"
        '{"name": "terminal", "arguments": {"command": "'
    )
    return {
        "task_id": f"trace-{idx}",
        "source_trace_id": f"group-{idx // 2}",
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_eval_gate_config(
    *,
    config_path: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
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
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _build_workspace(workdir: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="hermes_eval_gate_smoke_")
        return Path(tempdir.name), tempdir

    resolved = workdir.resolve()
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved, None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tiny held-out eval-gate run for CI and local smoke checks.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="optional persistent workspace; omitted uses a temporary directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    workspace, tempdir = _build_workspace(args.workdir)

    try:
        dataset_path = workspace / "traces.jsonl"
        checkpoint_dir = workspace / "checkpoints"
        output_dir = workspace / "eval_gate_out"
        config_path = workspace / "eval_gate_smoke.yaml"

        _write_jsonl(dataset_path, [_trace_row(i) for i in range(4)])

        model_path = checkpoint_dir / "iter_00001" / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        backend = TinyCausalLMBackend(
            TinyBackendConfig(seed=99, dim=16, n_heads=2, n_layers=1, max_len=512)
        )
        torch.save(backend.model.state_dict(), model_path)

        _write_eval_gate_config(
            config_path=config_path,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )

        command = [
            sys.executable,
            "-m",
            "hermes_agentic_rl.cli.main",
            "eval-gate",
            "--config",
            str(config_path),
        ]
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 3:
            raise RuntimeError(
                "expected eval-gate to return 3 for the CI hold-path smoke run; "
                f"got {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        summary_path = output_dir / "eval_summary.json"
        promotion_path = output_dir / "promotion.md"
        if not summary_path.exists():
            raise RuntimeError(f"missing eval summary: {summary_path}")
        if not promotion_path.exists():
            raise RuntimeError(f"missing promotion markdown: {promotion_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        promotion = promotion_path.read_text(encoding="utf-8")

        if summary.get("command") != "eval-gate":
            raise RuntimeError(f"unexpected command field: {summary.get('command')!r}")
        readout = summary.get("promotion_readout") or {}
        if readout.get("recommendation") != "hold":
            raise RuntimeError(
                "expected hold recommendation for the deterministic smoke gate; "
                f"got {readout.get('recommendation')!r}"
            )
        if readout.get("gate", {}).get("fail_on_hold") is not True:
            raise RuntimeError("eval-gate smoke did not force fail_on_hold=true")
        if "Recommendation: `hold`" not in promotion:
            raise RuntimeError("promotion.md did not record the expected hold recommendation")

        print(f"eval-gate smoke passed: {summary_path}")
        print(f"workspace: {workspace}")
        return 0
    finally:
        if tempdir is not None:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
