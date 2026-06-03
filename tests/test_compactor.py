import torch

from neural_kv.compactor import StillCompactor, StillLayerCompactor


def test_layer_compactor_shapes_and_initial_copy_bias() -> None:
    layer = StillLayerCompactor(head_dim=8, num_latents=4, rope_theta=10000.0)
    keys = torch.randn(2, 3, 16, 8)
    values = torch.randn(2, 3, 16, 8)
    compact_keys, compact_values, beta = layer(keys, values)
    assert compact_keys.shape == (2, 3, 4, 8)
    assert compact_values.shape == (2, 3, 4, 8)
    assert beta.shape == (2, 3, 4)


def test_full_compactor_returns_cache() -> None:
    compactor = StillCompactor(
        num_hidden_layers=2,
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
    )
    past = tuple((torch.randn(1, 2, 12, 8), torch.randn(1, 2, 12, 8)) for _ in range(2))
    cache = compactor(past, metadata={"source_tokens": 12})
    assert cache.num_layers == 2
    assert cache.num_tokens == 4
    assert cache.metadata["source_tokens"] == 12
