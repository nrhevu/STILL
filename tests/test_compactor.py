import math

import torch

from neural_kv.compactor import PerceiverBlock, StillCompactor, StillLayerCompactor


def test_layer_compactor_shapes_and_initial_copy_bias() -> None:
    layer = StillLayerCompactor(head_dim=8, num_latents=4, rope_theta=10000.0)
    keys = torch.randn(2, 3, 16, 8)
    values = torch.randn(2, 3, 16, 8)
    compact_keys, compact_values, beta = layer(keys, values)
    assert compact_keys.shape == (2, 3, 4, 8)
    assert compact_values.shape == (2, 3, 4, 8)
    assert beta.shape == (2, 3, 4)


def test_layer_compactor_accepts_rope_mode_none() -> None:
    layer = StillLayerCompactor(
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
        rope_mode="none",
    )
    keys = torch.randn(1, 2, 12, 8)
    values = torch.randn(1, 2, 12, 8)

    compact_keys, compact_values, beta = layer(keys, values)

    assert layer.rope_mode == "none"
    assert compact_keys.shape == (1, 2, 4, 8)
    assert compact_values.shape == (1, 2, 4, 8)
    assert beta.shape == (1, 2, 4)


def test_layer_compactor_rejects_invalid_rope_mode() -> None:
    try:
        StillLayerCompactor(
            head_dim=8,
            num_latents=4,
            rope_theta=10000.0,
            rope_mode="mrope",
        )
    except ValueError as exc:
        assert "rope_mode" in str(exc)
    else:
        raise AssertionError("expected invalid rope_mode to fail")


def test_layer_compactor_zero_beta_base_initializes_zero_bias() -> None:
    layer = StillLayerCompactor(
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
        beta_base="zero",
    )
    keys = torch.randn(1, 1, 16, 8)
    values = torch.randn(1, 1, 16, 8)

    _, _, beta = layer(keys, values)

    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)


def test_layer_compactor_beta_init_sets_initial_bias() -> None:
    layer = StillLayerCompactor(
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
        beta_base="zero",
        beta_init=-8.0,
    )
    keys = torch.randn(1, 1, 16, 8)
    values = torch.randn(1, 1, 16, 8)

    _, _, beta = layer(keys, values)

    assert torch.allclose(beta, torch.full_like(beta, -8.0), atol=1e-6)


def test_layer_compactor_log_beta_base_matches_compression_ratio() -> None:
    layer = StillLayerCompactor(
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
        beta_base="log_compression",
    )
    keys = torch.randn(1, 1, 16, 8)
    values = torch.randn(1, 1, 16, 8)

    _, _, beta = layer(keys, values)

    expected = torch.full_like(beta, math.log(4.0))
    assert torch.allclose(beta, expected, atol=1e-6)


def test_perceiver_identity_path_preserves_selected_kv_magnitude() -> None:
    block = PerceiverBlock(dim=4, rope_theta=10000.0, active_identity_path=True)
    latents = torch.zeros(1, 1, 4)
    kv_input = torch.tensor([[[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]]])
    positions = torch.zeros(1, dtype=torch.long)
    token_positions = torch.zeros(2, dtype=torch.long)

    output = block(
        latents,
        kv_input,
        latent_positions=positions,
        token_positions=token_positions,
    )

    assert torch.allclose(output, kv_input.mean(dim=1, keepdim=True), atol=1e-5)


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


def test_head_specific_latents_return_standard_cache_shape() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=4,
        num_latents=3,
        rope_theta=10000.0,
        num_key_value_heads=2,
        head_specific_latents=True,
    )
    keys = torch.randn(1, 2, 12, 4)
    values = torch.randn(1, 2, 12, 4)

    cache = compactor(((keys, values),))

    assert compactor.layers[0].latents.shape == (2, 3, 8)
    assert cache.keys[0].shape == (1, 2, 3, 4)
    assert cache.values[0].shape == (1, 2, 3, 4)
    assert cache.biases[0].shape == (1, 2, 3)


def test_head_specific_latents_reject_wrong_head_count() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=4,
        num_latents=3,
        rope_theta=10000.0,
        num_key_value_heads=2,
        head_specific_latents=True,
    )
    keys = torch.randn(1, 3, 12, 4)
    values = torch.randn(1, 3, 12, 4)

    try:
        compactor(((keys, values),))
    except ValueError as exc:
        assert "KV heads" in str(exc)
    else:
        raise AssertionError("expected wrong KV head count to fail")


def test_full_compactor_can_prepend_exact_sink_tokens() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
        sink_tokens=3,
    )
    keys = torch.randn(1, 2, 12, 8)
    values = torch.randn(1, 2, 12, 8)

    cache = compactor(((keys, values),))

    assert cache.num_tokens == 7
    assert torch.allclose(cache.keys[0][..., :3, :], keys[..., :3, :])
    assert torch.allclose(cache.values[0][..., :3, :], values[..., :3, :])
    assert torch.allclose(cache.biases[0][..., :3], torch.zeros_like(cache.biases[0][..., :3]))
    assert cache.metadata["sink_tokens"] == 3
    assert cache.metadata["latent_tokens"] == 4


def test_full_compactor_can_prepend_even_exact_tokens_after_sink() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        num_latents=2,
        rope_theta=10000.0,
        sink_tokens=1,
        exact_tokens=3,
        exact_strategy="even",
    )
    keys = torch.arange(1 * 1 * 10 * 2, dtype=torch.float32).reshape(1, 1, 10, 2)
    values = keys + 100.0

    cache = compactor(((keys, values),))

    assert cache.num_tokens == 6
    assert torch.allclose(cache.keys[0][..., :1, :], keys[..., :1, :])
    assert torch.allclose(cache.keys[0][..., 1:4, :], keys[..., [1, 5, 9], :])
    assert torch.allclose(cache.values[0][..., 1:4, :], values[..., [1, 5, 9], :])
    assert torch.allclose(cache.biases[0][..., :4], torch.zeros_like(cache.biases[0][..., :4]))
    assert cache.metadata["sink_tokens"] == 1
    assert cache.metadata["exact_tokens"] == 3
    assert cache.metadata["exact_strategy"] == "even"
    assert cache.metadata["latent_tokens"] == 2


def test_full_compactor_keeps_exact_bias_zero_with_negative_latents() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        num_latents=2,
        rope_theta=10000.0,
        beta_base="zero",
        beta_init=-8.0,
        exact_tokens=2,
        exact_strategy="even",
    )
    keys = torch.arange(1 * 1 * 8 * 2, dtype=torch.float32).reshape(1, 1, 8, 2)
    values = keys + 100.0

    cache = compactor(((keys, values),))

    assert cache.num_tokens == 4
    assert torch.allclose(cache.keys[0][..., :2, :], keys[..., [0, 7], :])
    assert torch.allclose(cache.values[0][..., :2, :], values[..., [0, 7], :])
    assert torch.allclose(cache.biases[0][..., :2], torch.zeros_like(cache.biases[0][..., :2]))
    assert torch.allclose(
        cache.biases[0][..., 2:],
        torch.full_like(cache.biases[0][..., 2:], -8.0),
    )
    assert cache.metadata["exact_tokens"] == 2
    assert cache.metadata["latent_tokens"] == 2


def test_full_compactor_can_boost_exact_token_bias() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        num_latents=1,
        rope_theta=10000.0,
        beta_base="zero",
        beta_init=-8.0,
        exact_tokens=2,
        exact_strategy="even",
        exact_beta=8.0,
    )
    keys = torch.arange(1 * 1 * 8 * 2, dtype=torch.float32).reshape(1, 1, 8, 2)
    values = keys + 100.0

    cache = compactor(((keys, values),))

    assert torch.allclose(cache.biases[0][..., :2], torch.full_like(cache.biases[0][..., :2], 8.0))
    assert torch.allclose(cache.biases[0][..., 2:], torch.full_like(cache.biases[0][..., 2:], -8.0))
    assert cache.metadata["exact_beta"] == 8.0


def test_full_compactor_can_select_kv_norm_exact_tokens_per_head() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        num_latents=1,
        rope_theta=10000.0,
        exact_tokens=2,
        exact_strategy="kv_norm",
    )
    keys = torch.zeros(1, 2, 5, 2)
    values = torch.zeros_like(keys)
    keys[0, 0, 3, 0] = 10.0
    values[0, 0, 1, 0] = 9.0
    keys[0, 1, 4, 0] = 8.0
    values[0, 1, 2, 0] = 7.0

    cache = compactor(((keys, values),))

    assert cache.num_tokens == 3
    assert torch.allclose(cache.keys[0][0, 0, :2], keys[0, 0, [1, 3]])
    assert torch.allclose(cache.values[0][0, 0, :2], values[0, 0, [1, 3]])
    assert torch.allclose(cache.keys[0][0, 1, :2], keys[0, 1, [2, 4]])
    assert torch.allclose(cache.values[0][0, 1, :2], values[0, 1, [2, 4]])
    assert torch.allclose(cache.biases[0][..., :2], torch.zeros_like(cache.biases[0][..., :2]))


def test_full_compactor_accepts_explicit_lexical_exact_indices() -> None:
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        num_latents=1,
        rope_theta=10000.0,
        exact_tokens=2,
        exact_strategy="lexical",
    )
    keys = torch.arange(1 * 1 * 6 * 2, dtype=torch.float32).reshape(1, 1, 6, 2)
    values = keys + 100.0

    cache = compactor(((keys, values),), exact_token_indices=torch.tensor([4, 1, 3]))

    assert cache.num_tokens == 3
    assert torch.allclose(cache.keys[0][..., :2, :], keys[..., [1, 3], :])
    assert torch.allclose(cache.values[0][..., :2, :], values[..., [1, 3], :])
    assert cache.metadata["exact_strategy"] == "lexical"


def test_grouped_compactor_reuses_depth_groups() -> None:
    compactor = StillCompactor(
        num_hidden_layers=4,
        head_dim=8,
        num_latents=4,
        rope_theta=10000.0,
        layer_compactor_groups=2,
    )
    past = tuple((torch.randn(1, 2, 12, 8), torch.randn(1, 2, 12, 8)) for _ in range(4))

    cache = compactor(past)

    assert len(compactor.layers) == 2
    assert compactor.num_hidden_layers == 4
    assert cache.num_layers == 4
    assert compactor._layer_group_index(0) == 0
    assert compactor._layer_group_index(1) == 0
    assert compactor._layer_group_index(2) == 1
    assert compactor._layer_group_index(3) == 1


def test_grouped_compactor_rejects_too_many_groups() -> None:
    try:
        StillCompactor(
            num_hidden_layers=2,
            head_dim=8,
            num_latents=4,
            rope_theta=10000.0,
            layer_compactor_groups=3,
        )
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("expected too many layer compactor groups to fail")
