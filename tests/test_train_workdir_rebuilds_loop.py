import json
import re
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_rebuilds_loop_per_item_when_workdir_base_is_set(
    tmp_path: Path,
    monkeypatch,
):
    # If the runtime loop caches cwd at init time, reusing it across items would write files
    # into the wrong directory after chdir. Train should rebuild loop per item when using
    # workdir_base to guarantee isolation.

    workdir_base = tmp_path / "workdirs"
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-1",
                        "instruction": "Create hello.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "hello.txt", "equals": "hello"}],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-2",
                        "instruction": "Create todo.txt and write buy milk",
                        "expected_output": "done",
                        "expected_files": [{"path": "todo.txt", "equals": "buy milk"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    export_path = tmp_path / "train.jsonl"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "  components:\n"
            "    - name: filesystem_verifier_reward\n"
            "      weight: 1.0\n"
            "trainer:\n"
            f"  export_training_path: {export_path}\n"
            f"  workdir_base: {workdir_base}\n"
        ),
        encoding="utf-8",
    )

    # Runtime loop that caches cwd at init time
    class CachingCwdLoop:
        def __init__(self):
            self.base = Path.cwd()

        async def run(self, prompt: str):
            m = re.search(r"Create\s+(\S+)\s+and\s+write\s+(.+)$", prompt)
            path = m.group(1)
            content = m.group(2)
            (self.base / path).parent.mkdir(parents=True, exist_ok=True)
            (self.base / path).write_text(content, encoding="utf-8")
            return {
                "messages": [{"role": "assistant", "content": "done"}],
                "tool_calls": [[{"function": {"name": "write_file", "arguments": "{}"}}]],
                "tool_results": [[{"role": "tool", "name": "write_file", "content": "{}"}]],
                "final_output": "done",
                "finished_naturally": True,
                "turns_used": 1,
                "metadata": {"runtime": "fake", "prompt": prompt},
            }

    def build_loop(_self, _config):
        return CachingCwdLoop()

    monkeypatch.setattr(
        "hermes_agentic_rl.runtime.fake_adapter.FakeRuntimeAdapter.build_agent_loop",
        build_loop,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train",
            "--config",
            str(config_path),
            "--limit",
            "2",
            "--overwrite",
            "--min-verifier-pass-ratio",
            "1.0",
        ],
    )

    assert main() == 0
