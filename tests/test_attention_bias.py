import torch

from neural_kv.attention_bias import merge_still_bias


class _Layer:
    layer_idx = 0
    num_key_value_groups = 2

    class config:
        num_attention_heads = 4


def test_merge_still_bias_repeats_kv_heads_and_pads() -> None:
    hidden_states = torch.zeros(1, 3, 8)
    attention_mask = torch.zeros(1, 1, 3, 8)
    biases = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])]

    merged = merge_still_bias(
        layer_module=_Layer(),
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        layer_biases=biases,
    )

    assert merged is not None
    assert merged.shape == (1, 4, 3, 8)
    assert torch.allclose(merged[0, 0, :, :2], torch.tensor([[1.0, 2.0]]).expand(3, 2))
    assert torch.allclose(merged[0, 1, :, :2], torch.tensor([[1.0, 2.0]]).expand(3, 2))
    assert torch.allclose(merged[0, 2, :, :2], torch.tensor([[3.0, 4.0]]).expand(3, 2))
    assert torch.allclose(merged[0, 3, :, :2], torch.tensor([[3.0, 4.0]]).expand(3, 2))
    assert torch.count_nonzero(merged[..., 2:]) == 0


def test_merge_still_bias_converts_bool_mask() -> None:
    hidden_states = torch.zeros(1, 1, 8)
    attention_mask = torch.tensor([[[[True, True, False]]]])
    biases = [torch.tensor([[[0.5, -0.25]]])]
    merged = merge_still_bias(
        layer_module=_Layer(),
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        layer_biases=biases,
    )
    assert merged is not None
    assert torch.allclose(merged[0, 0, 0, :2], torch.tensor([0.5, -0.25]))
    assert merged[0, 0, 0, 2] < -1e20
