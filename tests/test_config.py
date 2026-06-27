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


def test_vlm_qwen3_vl_config_uses_last_four_gpu_setup_and_gates() -> None:
    config = load_config(Path("config/experiment/vlm_qwen3_vl_8b_scienceqa_4gpu.yaml"))

    assert config["model"]["name"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert config["trainer"]["devices"] == 4
    assert config["trainer"]["strategy"] == "ddp"
    compactor = config["model"]["compactor"]
    assert compactor["num_latents"] == 1024
    assert compactor["sink_tokens"] == 8
    assert compactor["exact_tokens"] == 128
    assert compactor["exact_strategy"] == "kv_norm"
    assert compactor["rope_mode"] == "none"
    gates = config["gates"]
    assert [gate["min_full_accuracy"] for gate in gates] == [0.75, 0.60, 0.55]
    assert [gate["min_rows"] for gate in gates] == [256, 150, 847]
    assert gates[2]["summary_file"].endswith("mmmu_validation_generation_qwen_mmmu_lmms_cli.json")
