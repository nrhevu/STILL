from pathlib import Path

import torch

from neural_kv.models.compactor import StillCompactor
from neural_kv.training.initialization import load_initial_compactor_state


def test_load_initial_compactor_state_validates_and_returns_step(tmp_path: Path) -> None:
    config = {
        "num_latents": 2,
        "sink_tokens": 0,
        "exact_tokens": 0,
        "exact_strategy": "prefix",
        "num_blocks": 1,
        "latent_dropout": 0.0,
        "beta_base": "zero",
        "layer_compactor_groups": 0,
        "head_specific_latents": False,
    }
    source = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        rope_theta=10000.0,
        **config,
    )
    target = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        rope_theta=10000.0,
        **config,
    )
    with torch.no_grad():
        source.layers[0].beta_head.bias.fill_(3.0)
    checkpoint_path = tmp_path / "step_300.pt"
    torch.save(
        {
            "step": 300,
            "model": "unit-model",
            "context_length": 128,
            **config,
            "state_dict": source.state_dict(),
        },
        checkpoint_path,
    )

    step = load_initial_compactor_state(
        compactor=target,
        checkpoint_path=checkpoint_path,
        model_name="unit-model",
        context_length=128,
        compactor_config=config,
    )

    assert step == 300
    assert torch.equal(target.layers[0].beta_head.bias, source.layers[0].beta_head.bias)
