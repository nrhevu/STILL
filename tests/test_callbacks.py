from types import SimpleNamespace

import torch

from neural_kv.training.callbacks import LegacyCheckpointCallback


def test_legacy_checkpoint_callback_uses_initial_step_offset() -> None:
    callback = LegacyCheckpointCallback(output_dir="/tmp/neural-kv-test", save_every=100)
    trainer = SimpleNamespace(global_step=500, callback_metrics={})
    module = SimpleNamespace(initial_step=300)

    assert callback._legacy_step(trainer, module) == 800


def test_legacy_checkpoint_payload_contains_train_still_fields() -> None:
    callback = LegacyCheckpointCallback(output_dir="/tmp/neural-kv-test", save_every=100)
    del callback
    compactor = SimpleNamespace(state_dict=lambda: {"weight": torch.ones(1)})
    hparams = SimpleNamespace(
        compactor={
            "num_latents": 1024,
            "sink_tokens": 0,
            "exact_tokens": 0,
            "exact_strategy": "prefix",
            "num_blocks": 2,
            "layer_compactor_groups": 0,
            "head_specific_latents": False,
            "beta_base": "zero",
            "latent_dropout": 0.0,
        },
        model_name="Qwen/Qwen3-4B",
        context_length=8192,
        batch_size=2,
        kl_weight=1.0,
        reverse_kl_weight=0.5,
        ce_weight=0.1,
        aux_letter_loss_weight=0.05,
        aux_letter_enable_thinking=False,
        loss_mode="token",
        target_mode="teacher_response",
        score_mode="letter",
        eval_max_new_tokens=192,
        init_checkpoint="step_300.pt",
        use_chat_template=True,
        enable_thinking=True,
        eval_enable_thinking=False,
        balanced_answer_sampling=True,
        trainable_scope="all",
    )
    module = SimpleNamespace(hparams=hparams, compactor=compactor)

    payload = LegacyCheckpointCallback._checkpoint_payload(module, step=800, metrics={"loss": 1.0})

    assert payload["step"] == 800
    assert payload["model"] == "Qwen/Qwen3-4B"
    assert payload["num_latents"] == 1024
    assert payload["balanced_answer_sampling"] is True
    assert payload["state_dict"]["weight"].item() == 1.0
