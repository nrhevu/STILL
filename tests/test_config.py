from pathlib import Path

from neural_kv.utils.config import load_config


def test_load_config_merges_experiment_with_base() -> None:
    config = load_config(Path("config/experiment/smoke_tiny_llama.yaml"))

    assert config["model"]["compactor"]["num_blocks"] == 2
    assert config["model"]["compactor"]["num_latents"] == 16
    assert config["trainer"]["precision"] == "32-true"
