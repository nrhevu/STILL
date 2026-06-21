import torch

from neural_kv.models.checkpointing import parse_compactor_checkpoint


def test_parse_legacy_compactor_checkpoint() -> None:
    checkpoint = {
        "model": "base-model",
        "context_length": 128,
        "num_latents": 4,
        "state_dict": {"layers.0.latents": torch.zeros(4, 8)},
    }

    spec = parse_compactor_checkpoint(checkpoint)

    assert spec.model_name == "base-model"
    assert spec.context_length == 128
    assert spec.compactor["num_latents"] == 4
    assert "layers.0.latents" in spec.state_dict


def test_parse_lightning_compactor_checkpoint_strips_prefix() -> None:
    checkpoint = {
        "neural_kv": {
            "model": "base-model",
            "context_length": 256,
            "compactor": {"num_latents": 8},
        },
        "state_dict": {
            "compactor.layers.0.latents": torch.zeros(8, 16),
            "other.weight": torch.ones(1),
        },
    }

    spec = parse_compactor_checkpoint(checkpoint)

    assert spec.compactor["num_latents"] == 8
    assert "layers.0.latents" in spec.state_dict
    assert "other.weight" not in spec.state_dict
