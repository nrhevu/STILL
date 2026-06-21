from pathlib import Path

from neural_kv.utils.config import load_config


def test_sec_6k_reproduction_config_matches_documented_legacy_run() -> None:
    cfg = load_config(Path("config/experiment/sec_6k_qwen3_4b_8x_repro.yaml"))

    assert cfg["model"]["name"] == "Qwen/Qwen3-4B"
    assert cfg["model"]["context_length"] == 8192
    assert cfg["model"]["compactor"]["num_latents"] == 1024
    assert cfg["model"]["compactor"]["sink_tokens"] == 0
    assert cfg["model"]["compactor"]["exact_tokens"] == 0
    assert cfg["model"]["compactor"]["beta_base"] == "zero"
    assert cfg["model"]["compactor"]["head_specific_latents"] is False
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["learning_rate"] == 5.0e-6
    assert cfg["training"]["reverse_kl_weight"] == 0.5
    assert cfg["training"]["aux_letter_loss_weight"] == 0.05
    assert cfg["training"]["balanced_answer_sampling"] is True
    assert cfg["training"]["target_mode"] == "teacher_response"
    assert cfg["training"]["enable_thinking"] is True
    assert cfg["training"]["eval_enable_thinking"] is False
    assert cfg["training"]["init_checkpoint"].endswith("step_300.pt")
