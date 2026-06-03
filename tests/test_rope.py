import torch

from neural_kv.rope import apply_rope


def test_rope_inverse_roundtrip() -> None:
    x = torch.randn(2, 3, 5, 8)
    positions = torch.arange(5)
    rotated = apply_rope(x, positions, theta=10000.0)
    restored = apply_rope(rotated, positions, theta=10000.0, inverse=True)
    assert torch.allclose(restored, x, atol=1e-5)
