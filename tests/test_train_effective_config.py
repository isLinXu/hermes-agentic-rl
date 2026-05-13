import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_print_effective_config_only_outputs_json_with_cli_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    dataset_a = tmp_path / "a.jsonl"
    dataset_a.write_text("{}", encoding="utf-8")

    dataset_b = tmp_path / "b.jsonl"
    dataset_b.write_text("{}", encoding="utf-8")

    export_default = tmp_path / "outputs" / "default.jsonl"
    export_override = tmp_path / "outputs" / "override.jsonl"
    workdir_base = tmp_path / "outputs" / "workdirs" / "ec"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset_a}\n"
            "trainer:\n"
            f"  export_training_path: {export_default}\n"
            f"  workdir_base: {workdir_base}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_b),
            "--export",
            str(export_override),
            "--print-effective-config-only",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)

    assert payload["command"] == "train"
    assert payload["environment"]["dataset_path"].endswith("b.jsonl")
    assert payload["trainer"]["export_training_path"].endswith("override.jsonl")
    assert payload["trainer"]["workdir_base"].endswith(str(Path("outputs/workdirs/ec")))
