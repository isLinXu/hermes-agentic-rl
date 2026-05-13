import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def _write_minimal_atropos(path: Path) -> None:
    (path / "atroposlib").mkdir(parents=True, exist_ok=True)
    (path / "atroposlib" / "__init__.py").write_text("", encoding="utf-8")
    (path / "environments").mkdir(parents=True, exist_ok=True)


def _write_minimal_tinker_atropos(path: Path) -> None:
    (path / "tinker_atropos").mkdir(parents=True, exist_ok=True)
    (path / "tinker_atropos" / "__init__.py").write_text("", encoding="utf-8")
    (path / "tinker_atropos" / "config.py").write_text(
        "class TinkerAtroposConfig: pass\n", encoding="utf-8"
    )


def test_atropos_preflight_cli_outputs_json_and_fails_when_tinker_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ATROPOS_REPO", raising=False)
    monkeypatch.delenv("TINKER_ATROPOS_REPO", raising=False)
    # Preflight inspects current working directory for `atropos/` and `tinker-atropos/`.
    # Create minimal placeholders so we can test output shape without pulling huge repos.
    _write_minimal_atropos(tmp_path / "atropos")
    _write_minimal_tinker_atropos(tmp_path / "tinker-atropos")

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
    assert payload["atropos_source"] == "local"
    assert payload["tinker_atropos_source"] == "local"
    assert "checked_paths" in payload
    assert "missing" in payload
    assert exit_code == 1


def test_atropos_preflight_cli_discovers_subproject_layout(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ATROPOS_REPO", raising=False)
    monkeypatch.delenv("TINKER_ATROPOS_REPO", raising=False)
    _write_minimal_atropos(tmp_path / "subprojects" / "atropos")
    _write_minimal_tinker_atropos(
        tmp_path / "subprojects" / "hermes-agent" / "tinker-atropos"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "atropos-preflight"])

    exit_code = main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert payload["atropos_dir"] == str(tmp_path / "subprojects" / "atropos")
    assert payload["tinker_atropos_dir"] == str(
        tmp_path / "subprojects" / "hermes-agent" / "tinker-atropos"
    )
    assert payload["atropos_source"] == "subproject"
    assert payload["tinker_atropos_source"] == "subproject"
    assert exit_code == 1


def test_atropos_preflight_cli_prefers_env_over_default_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _write_minimal_atropos(tmp_path / "atropos")
    _write_minimal_tinker_atropos(tmp_path / "tinker-atropos")
    _write_minimal_atropos(tmp_path / "external" / "atropos")
    _write_minimal_tinker_atropos(tmp_path / "external" / "tinker-atropos")
    monkeypatch.setenv("ATROPOS_REPO", str(tmp_path / "external" / "atropos"))
    monkeypatch.setenv(
        "TINKER_ATROPOS_REPO", str(tmp_path / "external" / "tinker-atropos")
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "atropos-preflight"])

    exit_code = main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert payload["atropos_dir"] == str(tmp_path / "external" / "atropos")
    assert payload["tinker_atropos_dir"] == str(
        tmp_path / "external" / "tinker-atropos"
    )
    assert payload["atropos_source"] == "env"
    assert payload["tinker_atropos_source"] == "env"
    assert exit_code == 1
