from types import SimpleNamespace

import torch

from neural_kv.models.checkpointing import parse_compactor_checkpoint
from scripts.create_initial_compactor_checkpoint import build_checkpoint_payload


def test_build_initial_checkpoint_payload_preserves_8x_fields() -> None:
    model_config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10000.0,
    )
    compactor_config = {
        "num_latents": 3,
        "num_blocks": 1,
        "latent_dropout": 0.0,
        "beta_base": "zero",
        "beta_init": -8.0,
        "layer_compactor_groups": 1,
        "sink_tokens": 0,
        "exact_tokens": 1,
        "exact_strategy": "lexical",
        "exact_beta": 8.0,
        "head_specific_latents": True,
    }

    payload = build_checkpoint_payload(
        model_config,
        model_name="tiny-qwen",
        context_length=32,
        compactor_config=compactor_config,
        dtype=torch.bfloat16,
    )
    spec = parse_compactor_checkpoint(payload)

    assert payload["step"] == 0
    assert spec.model_name == "tiny-qwen"
    assert spec.context_length == 32
    assert spec.compactor["num_latents"] == 3
    assert spec.compactor["exact_tokens"] == 1
    assert spec.compactor["exact_strategy"] == "lexical"
    assert spec.compactor["exact_beta"] == 8.0
    assert spec.compactor["beta_init"] == -8.0
    assert spec.compactor["layer_compactor_groups"] == 1
    assert spec.state_dict["layers.0.beta_head.bias"].dtype == torch.bfloat16
    assert torch.allclose(
        spec.state_dict["layers.0.beta_head.bias"].float(),
        torch.full_like(spec.state_dict["layers.0.beta_head.bias"].float(), -8.0),
    )
