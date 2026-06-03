"""STILL-style neural KV-cache compactor."""

from __future__ import annotations

import math

import torch
from torch import nn

from neural_kv.cache import CompactKVCache, normalize_past_key_values
from neural_kv.rope import apply_rope, evenly_spaced_positions


class LatentSelfAttention(nn.Module):
    """Self-attention over compact latent slots."""

    def __init__(self, dim: int, *, zero_output: bool = True) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if zero_output:
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(x.shape[-1])
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        return self.out_proj(torch.matmul(attn, v))


class LatentCrossAttention(nn.Module):
    """RoPE-aware latent cross-attention into concatenated ``[K_unrotated; V]``."""

    def __init__(self, *, dim: int, rope_theta: float, active_identity_path: bool) -> None:
        super().__init__()
        self.dim = dim
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.reset_parameters(active_identity_path=active_identity_path)

    def reset_parameters(self, *, active_identity_path: bool) -> None:
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)
        direction = torch.ones(self.dim, dtype=self.q_proj.bias.dtype)
        direction = direction / direction.norm()
        with torch.no_grad():
            self.q_proj.bias.copy_(direction)
            self.k_proj.bias.copy_(10.0 * direction)
            nn.init.eye_(self.v_proj.weight)
            if active_identity_path:
                nn.init.eye_(self.out_proj.weight)
            else:
                nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        latents: torch.Tensor,
        kv_input: torch.Tensor,
        *,
        latent_positions: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        q = apply_rope(self.q_proj(latents), latent_positions, theta=self.rope_theta)
        k = apply_rope(self.k_proj(kv_input), token_positions, theta=self.rope_theta)
        v = self.v_proj(kv_input)
        scale = 1.0 / math.sqrt(self.dim)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        return self.out_proj(torch.matmul(weights, v))


class PerceiverBlock(nn.Module):
    """One STILL perceiver block: cross-attend, then coordinate latents."""

    def __init__(self, *, dim: int, rope_theta: float, active_identity_path: bool) -> None:
        super().__init__()
        self.cross_attn = LatentCrossAttention(
            dim=dim,
            rope_theta=rope_theta,
            active_identity_path=active_identity_path,
        )
        self.cross_norm = nn.RMSNorm(dim)
        self.self_attn = LatentSelfAttention(dim, zero_output=True)
        self.self_norm = nn.RMSNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        kv_input: torch.Tensor,
        *,
        latent_positions: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        latents = self.cross_norm(
            latents
            + self.cross_attn(
                latents,
                kv_input,
                latent_positions=latent_positions,
                token_positions=token_positions,
            )
        )
        return self.self_norm(latents + self.self_attn(latents))


class StillLayerCompactor(nn.Module):
    """Compress one transformer's layer cache to a fixed number of latent slots."""

    def __init__(
        self,
        *,
        head_dim: int,
        num_latents: int,
        rope_theta: float,
        num_blocks: int = 2,
        latent_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.head_dim = int(head_dim)
        self.num_latents = int(num_latents)
        self.latent_dim = self.head_dim * 2
        self.rope_theta = float(rope_theta)
        self.latent_dropout = float(latent_dropout)

        self.latents = nn.Parameter(torch.zeros(self.num_latents, self.latent_dim))
        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(
                    dim=self.latent_dim,
                    rope_theta=self.rope_theta,
                    active_identity_path=(idx == 0),
                )
                for idx in range(num_blocks)
            ]
        )
        self.key_head = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.value_head = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.beta_head = nn.Linear(self.latent_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.latents.zero_()
            self.key_head.weight.zero_()
            self.key_head.weight[:, : self.head_dim] = torch.eye(self.head_dim)
            self.value_head.weight.zero_()
            self.value_head.weight[:, self.head_dim :] = torch.eye(self.head_dim)
            self.beta_head.weight.zero_()
            self.beta_head.bias.zero_()

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if keys.dim() != 4:
            raise ValueError("keys and values must be shaped [batch, heads, tokens, dim]")
        batch, heads, seq_len, head_dim = keys.shape
        if head_dim != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head_dim}")

        dtype = keys.dtype
        module_dtype = self.key_head.weight.dtype
        token_positions = torch.arange(seq_len, device=keys.device, dtype=torch.long)
        latent_positions = evenly_spaced_positions(self.num_latents, seq_len, keys.device)

        unrotated_keys = apply_rope(keys, token_positions, theta=self.rope_theta, inverse=True)
        kv_input = torch.cat([unrotated_keys, values], dim=-1)
        kv_input = kv_input.reshape(batch * heads, seq_len, self.latent_dim).to(module_dtype)
        latents = self.latents.unsqueeze(0).expand(batch * heads, -1, -1).to(module_dtype)

        if self.training and self.latent_dropout > 0:
            keep = torch.rand(latents.shape[:2], device=latents.device) >= self.latent_dropout
            latents = latents * keep.unsqueeze(-1)

        for block in self.blocks:
            latents = block(
                latents,
                kv_input,
                latent_positions=latent_positions,
                token_positions=token_positions,
            )

        compact_keys = self.key_head(latents)
        compact_values = self.value_head(latents)
        beta = self.beta_head(latents).squeeze(-1)
        compact_keys = apply_rope(compact_keys, latent_positions, theta=self.rope_theta)

        compact_keys = compact_keys.reshape(batch, heads, self.num_latents, head_dim).to(dtype)
        compact_values = compact_values.reshape(batch, heads, self.num_latents, head_dim).to(dtype)
        beta = beta.reshape(batch, heads, self.num_latents).to(dtype)
        return compact_keys, compact_values, beta


class StillCompactor(nn.Module):
    """Per-layer STILL compactor for a frozen decoder-only transformer."""

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        head_dim: int,
        num_latents: int,
        rope_theta: float,
        num_blocks: int = 2,
        latent_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_latents = int(num_latents)
        self.layers = nn.ModuleList(
            [
                StillLayerCompactor(
                    head_dim=head_dim,
                    num_latents=num_latents,
                    rope_theta=rope_theta,
                    num_blocks=num_blocks,
                    latent_dropout=latent_dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    @classmethod
    def from_model_config(
        cls,
        config,
        *,
        num_latents: int,
        num_blocks: int = 2,
        latent_dropout: float = 0.0,
    ) -> StillCompactor:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        return cls(
            num_hidden_layers=int(config.num_hidden_layers),
            head_dim=int(head_dim),
            num_latents=num_latents,
            rope_theta=rope_theta,
            num_blocks=num_blocks,
            latent_dropout=latent_dropout,
        )

    def forward(
        self,
        past_key_values,
        *,
        metadata: dict[str, object] | None = None,
    ) -> CompactKVCache:
        normalized = normalize_past_key_values(past_key_values)
        if len(normalized) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} cache layers, got {len(normalized)}")
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for layer_compactor, (layer_keys, layer_values) in zip(
            self.layers,
            normalized,
            strict=True,
        ):
            compact_k, compact_v, beta = layer_compactor(layer_keys, layer_values)
            keys.append(compact_k)
            values.append(compact_v)
            biases.append(beta)
        return CompactKVCache(
            keys=keys,
            values=values,
            biases=biases,
            metadata=metadata,
            detach=False,
        )
