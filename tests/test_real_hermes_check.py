import subprocess
import types
from pathlib import Path

from scripts.check_real_hermes import (
    build_overall_status,
    check_adapter_build,
    check_ai_agent_symbol,
    check_entrypoint_detection,
    check_path_exists,
    check_python_version,
    check_run_agent_import,
    format_summary_lines,
    main,
    run_checks,
)


def test_check_python_version_flags_python310_as_error():
    result = check_python_version((3, 10, 12))

    assert result["status"] == "error"
    assert "3.11+" in result["details"]


def test_check_python_version_flags_python311_as_ok():
    result = check_python_version((3, 11, 9))

    assert result["status"] == "ok"


def test_check_path_exists_reports_missing_file(tmp_path: Path):
    result = check_path_exists(tmp_path / "missing.txt", label="sample_config")

    assert result["status"] == "warning"
    assert "missing" in result["details"].lower()


def test_build_overall_status_uses_error_over_warning():
    checks = {
        "python_version": {"status": "ok", "details": "ok"},
        "adapter_build": {"status": "error", "details": "broken"},
        "sample_dataset": {"status": "warning", "details": "missing"},
    }

    assert build_overall_status(checks) == "error"


def test_check_run_agent_import_reports_missing_module():
    result = check_run_agent_import(importer=lambda: (_ for _ in ()).throw(ImportError("missing run_agent")))

    assert result["status"] == "error"
    assert "missing run_agent" in result["details"]


def test_check_ai_agent_symbol_reports_ok_when_present():
    module = types.SimpleNamespace(AIAgent=object)

    result = check_ai_agent_symbol(module)

    assert result["status"] == "ok"
    assert "AIAgent" in result["details"]


def test_check_entrypoint_detection_reports_detected_entrypoint():
    entrypoint = types.SimpleNamespace(module_name="run_agent", attr_name="AIAgent")

    result = check_entrypoint_detection(detector=lambda: entrypoint)

    assert result["status"] == "ok"
    assert "run_agent:AIAgent" in result["details"]


def test_check_adapter_build_reports_error_when_python_too_old():
    result = check_adapter_build(
        version_info=(3, 10, 12),
        adapter_builder=lambda: None,
    )

    assert result["status"] == "error"
    assert "Python 3.11+" in result["details"]


def test_check_adapter_build_reports_ok_when_builder_succeeds():
    result = check_adapter_build(
        version_info=(3, 11, 9),
        adapter_builder=lambda: object(),
    )

    assert result["status"] == "ok"


def test_format_summary_lines_contains_status_prefixes():
    checks = {
        "python_version": {"status": "ok", "details": "Python version is 3.11.9"},
        "sample_dataset": {"status": "warning", "details": "sample dataset missing"},
    }

    lines = format_summary_lines(checks)

    assert any(line.startswith("[OK]") for line in lines)
    assert any(line.startswith("[WARN]") for line in lines)


def test_run_checks_builds_report_and_strips_imported_module(monkeypatch):
    import scripts.check_real_hermes as module

    run_agent_module = types.SimpleNamespace(AIAgent=object)

    monkeypatch.setattr(
        module,
        "check_python_version",
        lambda version_info: {"status": "ok", "details": f"Python version is {version_info[0]}.{version_info[1]}.{version_info[2]}"},
    )
    monkeypatch.setattr(
        module,
        "check_run_agent_import",
        lambda: {"status": "ok", "details": "run_agent importable", "module": run_agent_module},
    )
    monkeypatch.setattr(
        module,
        "check_ai_agent_symbol",
        lambda imported_module: {"status": "ok", "details": f"AIAgent symbol found: {imported_module is run_agent_module}"},
    )
    monkeypatch.setattr(
        module,
        "check_entrypoint_detection",
        lambda: {"status": "ok", "details": "detected entrypoint: run_agent:AIAgent"},
    )
    monkeypatch.setattr(
        module,
        "check_adapter_build",
        lambda version_info: {"status": "ok", "details": f"adapter build ok on {version_info[0]}.{version_info[1]}.{version_info[2]}"},
    )
    monkeypatch.setattr(
        module,
        "check_path_exists",
        lambda path, label: {"status": "warning" if label == "sample_dataset" else "ok", "details": f"{label}:{path.name}"},
    )

    report = run_checks()

    assert report["overall_status"] == "warning"
    assert report["checks"]["run_agent_import"]["status"] == "ok"
    assert "module" not in report["checks"]["run_agent_import"]
    assert report["checks"]["sample_dataset"]["status"] == "warning"


def test_main_prints_summary_and_json_and_returns_error(monkeypatch, capsys):
    import scripts.check_real_hermes as module

    monkeypatch.setattr(
        module,
        "run_checks",
        lambda: {
            "overall_status": "error",
            "checks": {
                "adapter_build": {"status": "error", "details": "broken"},
            },
        },
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "[ERROR] adapter_build: broken" in captured.out
    assert '"overall_status": "error"' in captured.out


def test_real_hermes_docs_exist():
    workspace_root = Path(__file__).resolve().parent.parent

    assert (workspace_root / "docs" / "real-hermes-check.md").exists()


def test_check_script_runs_from_repo_root_without_import_error():
    workspace_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        ["python3", "scripts/check_real_hermes.py"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )

    assert "ModuleNotFoundError: No module named 'hermes_agentic_rl'" not in result.stderr
    assert '"overall_status":' in result.stdout
