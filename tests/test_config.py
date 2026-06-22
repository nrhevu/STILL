from pathlib import Path

from neural_kv.utils.config import load_config


def test_load_config_merges_experiment_with_base() -> None:
    config = load_config(Path("config/experiment/smoke_tiny_llama.yaml"))

    assert config["model"]["compactor"]["num_blocks"] == 2
    assert config["model"]["compactor"]["num_latents"] == 16
    assert config["trainer"]["precision"] == "32-true"


def test_ruler_200k_qwen3_235b_config_targets_8x() -> None:
    config = load_config(Path("config/experiment/ruler_200k_qwen3_235b_2507_8x.yaml"))

    assert config["model"]["name"] == "Qwen/Qwen3-235B-A22B-Instruct-2507"
    compactor = config["model"]["compactor"]
    compact_tokens = compactor["num_latents"] + compactor["exact_tokens"]

    assert config["model"]["context_length"] == 200000
    assert compactor["num_latents"] == 23976
    assert compactor["exact_tokens"] == 1024
    assert compactor["exact_strategy"] == "lexical"
    assert compactor["exact_beta"] == 8.0
    assert compactor["beta_init"] == -8.0
    assert compactor["layer_compactor_groups"] == 1
    assert config["model"]["context_length"] / compact_tokens == 8
    assert config["training"]["target_mode"] == "teacher_response"
    assert config["training"]["eval_enable_thinking"] is False
    assert config["data"]["train_file"] == "data/ruler_200k/train.teacher.jsonl"
