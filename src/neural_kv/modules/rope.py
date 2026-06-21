"""RoPE utilities used by the STILL compactor."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the Llama/Qwen split-half rotary transform on the final dimension."""
    left = x[..., : x.shape[-1] // 2]
    right = x[..., x.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def rope_frequencies(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Return inverse frequency values for RoPE."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


def rope_cos_sin(
    positions: torch.Tensor,
    *,
    dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine tables for arbitrary absolute positions."""
    inv_freq = rope_frequencies(dim, theta, positions.device)
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or invert RoPE on the last dimension of ``x``."""
    cos, sin = rope_cos_sin(positions, dim=x.shape[-1], theta=theta)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if inverse:
        sin = -sin
    x_float = x.to(torch.float32)
    rotated = (x_float * cos) + (rotate_half(x_float) * sin)
    return rotated.to(x.dtype)


def evenly_spaced_positions(
    num_positions: int,
    source_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign compact latents evenly spaced positions over the source sequence."""
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")
    if num_positions == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    values = torch.linspace(
        0,
        max(source_length - 1, 0),
        steps=num_positions,
        device=device,
        dtype=torch.float32,
    )
    return values.round().to(torch.long)
