import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_agentic_rl.envs.terminal_task_env import TerminalTaskEnv  # noqa: E402


def build_env() -> TerminalTaskEnv:
    data = [
        {
            "task_id": "write-hello",
            "instruction": "Create hello.txt and write hello",
            "expected_output": "done",
        }
    ]
    return TerminalTaskEnv(dataset=data)


if __name__ == "__main__":
    env = build_env()
    print(env.format_prompt(env.dataset[0]))
