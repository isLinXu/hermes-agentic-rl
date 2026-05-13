"""Hermes runtime sanity check script."""
# ruff: noqa: I001, E402

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after sys.path injection so the script works from a plain checkout.
from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter
from hermes_agentic_rl.runtime.hermes_entrypoints import (
    find_hermes_entrypoint,
)
from hermes_agentic_rl.integrations.hermes_repo import prepare_hermes_imports


def check_python_version(version_info: tuple[int, int, int]) -> dict[str, str]:
    major, minor, patch = version_info
    if (major, minor) >= (3, 11):
        return {
            "status": "ok",
            "details": f"Python version is {major}.{minor}.{patch}",
        }
    return {
        "status": "error",
        "details": f"Python {major}.{minor}.{patch} is too old; hermes-agent requires Python 3.11+",
    }


def check_path_exists(path: Path, label: str) -> dict[str, str]:
    if path.exists():
        return {"status": "ok", "details": f"{label} exists: {path}"}
    return {"status": "warning", "details": f"{label} missing: {path}"}


def default_run_agent_importer() -> Any:
    prepare_hermes_imports(base_dir=PROJECT_ROOT)
    return importlib.import_module("run_agent")


def check_run_agent_import(importer: Callable[[], Any] | None = None) -> dict[str, str | Any]:
    importer = importer or default_run_agent_importer
    try:
        module = importer()
        return {"status": "ok", "details": "run_agent importable", "module": module}
    except Exception as exc:
        return {"status": "error", "details": str(exc), "module": None}


def check_ai_agent_symbol(module: Any) -> dict[str, str]:
    if module is None:
        return {"status": "error", "details": "run_agent module unavailable"}
    if hasattr(module, "AIAgent"):
        return {"status": "ok", "details": "AIAgent symbol found"}
    return {"status": "error", "details": "AIAgent symbol missing"}


def check_entrypoint_detection(detector: Callable[[], Any] | None = None) -> dict[str, str]:
    detector = detector or find_hermes_entrypoint
    try:
        entrypoint = detector()
        return {
            "status": "ok",
            "details": f"detected entrypoint: {entrypoint.module_name}:{entrypoint.attr_name}",
        }
    except Exception as exc:
        return {"status": "error", "details": str(exc)}


def default_adapter_builder() -> Any:
    return HermesRuntimeAdapter().build_agent_loop({"runtime": {"integration": "hermes"}})


def check_adapter_build(
    version_info: tuple[int, int, int],
    adapter_builder: Callable[[], Any] | None = None,
) -> dict[str, str]:
    major, minor, patch = version_info
    if (major, minor) < (3, 11):
        return {
            "status": "error",
            "details": f"adapter build skipped: Python {major}.{minor}.{patch} does not satisfy Python 3.11+",
        }

    adapter_builder = adapter_builder or default_adapter_builder
    try:
        adapter_builder()
        return {"status": "ok", "details": "Hermes runtime adapter build succeeded"}
    except Exception as exc:
        return {"status": "warning", "details": f"adapter build failed: {exc}"}


def build_overall_status(checks: dict[str, dict[str, str]]) -> str:
    statuses = [check["status"] for check in checks.values()]
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def format_summary_lines(checks: dict[str, dict[str, str]]) -> list[str]:
    mapping = {"ok": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}
    lines: list[str] = []
    for name, result in checks.items():
        prefix = mapping.get(result["status"], "[INFO]")
        lines.append(f"{prefix} {name}: {result['details']}")
    return lines


def run_checks() -> dict[str, Any]:
    workspace_root = PROJECT_ROOT
    repo_resolution = prepare_hermes_imports(base_dir=workspace_root)
    python_result = check_python_version(sys.version_info[:3])
    run_agent_result = check_run_agent_import()
    ai_agent_result = check_ai_agent_symbol(run_agent_result.get("module"))
    entrypoint_result = check_entrypoint_detection()
    adapter_result = check_adapter_build(sys.version_info[:3])
    config_result = check_path_exists(
        workspace_root / "configs" / "terminal_grpo_hermes.yaml",
        "sample_config",
    )
    dataset_result = check_path_exists(
        workspace_root / "data" / "minimal_terminal_tasks.jsonl",
        "sample_dataset",
    )
    repo_result = {
        "status": "ok" if repo_resolution.is_present and repo_resolution.has_run_agent else "warning",
        "details": (
            f"hermes repo: {repo_resolution.repo_path}"
            if repo_resolution.repo_path
            else "local hermes-agent subproject missing"
        ),
    }

    checks = {
        "python_version": python_result,
        "hermes_repo": repo_result,
        "run_agent_import": {k: v for k, v in run_agent_result.items() if k != "module"},
        "ai_agent_symbol": ai_agent_result,
        "entrypoint_detection": entrypoint_result,
        "adapter_build": adapter_result,
        "sample_config": config_result,
        "sample_dataset": dataset_result,
    }
    return {
        "overall_status": build_overall_status(checks),
        "checks": checks,
    }


def main() -> int:
    report = run_checks()
    for line in format_summary_lines(report["checks"]):
        print(line)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
