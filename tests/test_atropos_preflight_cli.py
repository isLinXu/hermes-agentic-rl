import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main
from hermes_agentic_rl.integrations import atropos_preflight


def test_atropos_preflight_cli_outputs_json_and_fails_when_tinker_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    # Preflight inspects current working directory for subproject checkouts.
    # Create minimal placeholders so we can test output shape without pulling huge repos.
    atropos_dir = tmp_path / "subprojects" / "atropos"
    tinker_dir = tmp_path / "subprojects" / "tinker-atropos"
    atropos_dir.mkdir(parents=True, exist_ok=True)
    (tinker_dir / "tinker_atropos").mkdir(parents=True, exist_ok=True)
    (tinker_dir / "tinker_atropos" / "__init__.py").write_text("", encoding="utf-8")
    (tinker_dir / "tinker_atropos" / "config.py").write_text(
        "class TinkerAtroposConfig: pass\n",
        encoding="utf-8",
    )
    (atropos_dir / "atroposlib").mkdir(parents=True, exist_ok=True)
    (atropos_dir / "atroposlib" / "__init__.py").write_text("", encoding="utf-8")

    available_modules = {
        "atroposlib",
        "tinker_atropos.config",
        "wandb",
        "torch",
        "transformers",
    }

    monkeypatch.setattr(
        atropos_preflight,
        "_module_exists",
        lambda name: name in available_modules,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "atropos-preflight"],
    )

    exit_code = main()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)

    assert payload["base_dir"]
    assert payload["atropos_dir"] is not None
    assert payload["tinker_atropos_dir"] is not None
    assert "missing" in payload
    assert payload["missing"] == ["python:tinker"]
    assert payload["missing_details"] == [
        {
            "kind": "python_module_missing",
            "module": "tinker",
            "required": True,
            "reason": "required for TinkerAtroposTrainer",
        }
    ]
    assert exit_code == 1


def test_atropos_preflight_cli_outputs_structured_json_for_missing_local_submodules(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(atropos_preflight, "_module_exists", lambda _name: False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "atropos-preflight"],
    )

    exit_code = main()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)

    assert payload["atropos_dir"] is None
    assert payload["tinker_atropos_dir"] is None
    assert payload["missing"] == [
        "local_dir:subprojects/atropos",
        "local_dir:subprojects/tinker-atropos",
        "python:atroposlib",
        "python:tinker_atropos.config",
        "python:tinker",
    ]
    assert payload["missing_details"] == [
        {
            "kind": "local_dir_missing",
            "repo": "atropos",
            "path": str((tmp_path / "subprojects" / "atropos").resolve()),
        },
        {
            "kind": "local_dir_missing",
            "repo": "tinker-atropos",
            "path": str((tmp_path / "subprojects" / "tinker-atropos").resolve()),
        },
        {
            "kind": "python_module_missing",
            "module": "atroposlib",
            "required": True,
            "reason": "required for local Atropos integration",
        },
        {
            "kind": "python_module_missing",
            "module": "tinker_atropos.config",
            "required": True,
            "reason": "required for local tinker-atropos config import",
        },
        {
            "kind": "python_module_missing",
            "module": "tinker",
            "required": True,
            "reason": "required for TinkerAtroposTrainer",
        },
    ]
    assert exit_code == 1


def test_atropos_preflight_cli_keeps_json_when_python_submodule_probe_raises(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    atropos_dir = tmp_path / "subprojects" / "atropos"
    tinker_dir = tmp_path / "subprojects" / "tinker-atropos"
    atropos_dir.mkdir(parents=True, exist_ok=True)
    tinker_dir.mkdir(parents=True, exist_ok=True)

    def fake_find_spec(name: str):
        if name == "tinker_atropos.config":
            raise ModuleNotFoundError("No module named 'tinker_atropos'")
        if name in {"atroposlib", "tinker", "wandb", "torch", "transformers"}:
            return object()
        return None

    monkeypatch.setattr(atropos_preflight.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "atropos-preflight"],
    )

    exit_code = main()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)

    assert payload["atropos_dir"] is not None
    assert payload["tinker_atropos_dir"] is not None
    assert payload["missing"] == ["python:tinker_atropos.config"]
    assert payload["missing_details"] == [
        {
            "kind": "python_module_missing",
            "module": "tinker_atropos.config",
            "required": True,
            "reason": "required for local tinker-atropos config import",
        }
    ]
    assert exit_code == 1
