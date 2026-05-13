import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


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
    assert exit_code == 1
