import importlib
import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main
from hermes_agentic_rl.integrations.hermes_preflight import run_hermes_preflight
from hermes_agentic_rl.integrations.hermes_repo import (
    HERMES_AGENT_REPO_ENV,
    resolve_hermes_repo,
)


def _write_fake_hermes_repo(repo_path: Path) -> None:
    (repo_path / "environments").mkdir(parents=True, exist_ok=True)
    (repo_path / "tools").mkdir(parents=True, exist_ok=True)
    (repo_path / "environments" / "__init__.py").write_text("", encoding="utf-8")
    (repo_path / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (repo_path / "environments" / "agent_loop.py").write_text(
        (
            "class HermesAgentLoop:\n"
            "    async def run(self, prompt):\n"
            "        return {\n"
            "            'messages': [{'role': 'assistant', 'content': 'done'}],\n"
            "            'tool_calls': [],\n"
            "            'tool_results': [],\n"
            "            'final_output': 'done',\n"
            "            'finished_naturally': True,\n"
            "            'turns_used': 1,\n"
            "        }\n"
        ),
        encoding="utf-8",
    )
    (repo_path / "run_agent.py").write_text(
        (
            "class AIAgent:\n"
            "    def __init__(self, **kwargs):\n"
            "        self.kwargs = kwargs\n"
            "\n"
            "    def run_conversation(self, user_message, task_id=None, system_message=None):\n"
            "        return {\n"
            "            'final_response': 'done',\n"
            "            'messages': [\n"
            "                {'role': 'user', 'content': user_message},\n"
            "                {'role': 'assistant', 'content': 'done'},\n"
            "            ],\n"
            "            'task_id': task_id,\n"
            "            'turns_used': 1,\n"
            "        }\n"
        ),
        encoding="utf-8",
    )


def test_resolve_hermes_repo_prefers_config_over_env_and_subproject(tmp_path: Path, monkeypatch):
    configured_repo = tmp_path / "configured-hermes"
    configured_repo.mkdir()
    env_repo = tmp_path / "env-hermes"
    env_repo.mkdir()
    (tmp_path / "subprojects" / "hermes-agent").mkdir(parents=True)
    monkeypatch.setenv(HERMES_AGENT_REPO_ENV, str(env_repo))

    resolution = resolve_hermes_repo(
        {"repo_path": str(configured_repo)},
        base_dir=tmp_path,
    )

    assert resolution.source == "config"
    assert resolution.repo_path == configured_repo.resolve()


def test_cli_rollout_works_with_local_repo_path_even_if_availability_probe_is_false(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    fake_repo = tmp_path / "fake-hermes-agent"
    fake_repo.mkdir()
    _write_fake_hermes_repo(fake_repo)

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create x.txt",
                "expected_output": "done",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: hermes\n"
            f"  repo_path: {fake_repo}\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path / 'train.jsonl'}\n"
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "trajectory.json"

    monkeypatch.delitem(importlib.sys.modules, "run_agent", raising=False)
    monkeypatch.delitem(importlib.sys.modules, "environments", raising=False)
    monkeypatch.delitem(importlib.sys.modules, "environments.agent_loop", raising=False)
    monkeypatch.setattr(
        "hermes_agentic_rl.runtime.hermes_adapter.HermesRuntimeAdapter.is_available",
        lambda self: False,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "rollout",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "trajectory saved" in captured.out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["runtime"]["runtime"] == "hermes"
    assert payload["final_output"] == "done"


def test_hermes_preflight_detects_local_subproject(tmp_path: Path):
    fake_repo = tmp_path / "subprojects" / "hermes-agent"
    fake_repo.mkdir(parents=True)
    _write_fake_hermes_repo(fake_repo)

    result = run_hermes_preflight(tmp_path)

    assert result.repo_source == "subproject"
    assert result.repo_path == fake_repo.resolve()
    assert "python:run_agent" not in result.missing
    assert "python:environments.agent_loop" not in result.missing


def test_run_hermes_preflight_reports_probe_errors_as_structured_missing(
    tmp_path: Path,
    monkeypatch,
):
    fake_repo = tmp_path / "subprojects" / "hermes-agent"
    fake_repo.mkdir(parents=True)
    _write_fake_hermes_repo(fake_repo)
    (fake_repo / "environments" / "__init__.py").write_text(
        "raise RuntimeError('boom')\n",
        encoding="utf-8",
    )

    monkeypatch.delitem(importlib.sys.modules, "run_agent", raising=False)
    monkeypatch.delitem(importlib.sys.modules, "environments", raising=False)
    monkeypatch.delitem(importlib.sys.modules, "environments.agent_loop", raising=False)

    result = run_hermes_preflight(tmp_path)

    assert result.repo_source == "subproject"
    assert result.python_ok["environments.agent_loop"] is False
    assert "python_probe_error:environments.agent_loop:RuntimeError" in result.missing


def test_cli_hermes_preflight_command_prints_json(tmp_path: Path, monkeypatch, capsys):
    fake_repo = tmp_path / "subprojects" / "hermes-agent"
    fake_repo.mkdir(parents=True)
    _write_fake_hermes_repo(fake_repo)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "hermes-preflight"])

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"repo_source": "subproject"' in captured.out


def test_run_hermes_preflight_returns_structured_result_for_repo_root():
    workspace_root = Path(__file__).resolve().parent.parent

    result = run_hermes_preflight(workspace_root)
    payload = result.as_dict()

    assert isinstance(result.python_ok, dict)
    assert isinstance(result.missing, list)
    assert result.repo_source in {None, "config", "env", "subproject"}
    assert payload["base_dir"] == str(workspace_root.resolve())
    assert payload["repo_source"] == result.repo_source
    assert payload["repo_path"] == (str(result.repo_path) if result.repo_path else None)
    assert payload["python_ok"] == result.python_ok
    assert payload["missing"] == result.missing
