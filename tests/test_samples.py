from pathlib import Path

from hermes_agentic_rl.config import load_config
from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset


def test_sample_dataset_and_hermes_config_exist():
    workspace_root = Path(__file__).resolve().parent.parent
    dataset_path = workspace_root / "data" / "minimal_terminal_tasks.jsonl"
    config_path = workspace_root / "configs" / "terminal_grpo_hermes.yaml"

    assert dataset_path.exists()
    assert config_path.exists()

    items = load_jsonl_dataset(dataset_path)
    config = load_config(config_path)

    assert len(items) >= 1
    assert items[0]["instruction"]
    assert config["runtime"]["integration"] == "hermes"
    assert "terminal" in config["runtime"]["enabled_toolsets"]
    assert config["trainer"]["workdir_base"]
