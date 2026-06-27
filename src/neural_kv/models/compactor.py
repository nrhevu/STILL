"""STILL-style neural KV-cache compactor."""

from __future__ import annotations

import math

import torch
from torch import nn

from neural_kv.modules.cache import CompactKVCache, normalize_past_key_values
from neural_kv.modules.rope import apply_rope, evenly_spaced_positions

EXACT_TOKEN_STRATEGIES = {"prefix", "even", "kv_norm", "lexical"}
ROPE_MODES = {"default", "none"}


def _apply_rope_mode(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
    rope_mode: str,
    inverse: bool = False,
) -> torch.Tensor:
    if rope_mode == "none":
        return x
    if rope_mode == "default":
        return apply_rope(x, positions, theta=theta, inverse=inverse)
    raise ValueError(f"Unsupported rope_mode: {rope_mode}")


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

    def __init__(
        self,
        *,
        dim: int,
        rope_theta: float,
        rope_mode: str,
        active_identity_path: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rope_theta = rope_theta
        self.rope_mode = rope_mode
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.reset_parameters(active_identity_path=active_identity_path)

    def reset_parameters(self, *, active_identity_path: bool) -> None:
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)
        direction = torch.ones(self.dim, dtype=self.q_proj.bias.dtype)
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
        q = _apply_rope_mode(
            self.q_proj(latents),
            latent_positions,
            theta=self.rope_theta,
            rope_mode=self.rope_mode,
        )
        k = _apply_rope_mode(
            self.k_proj(kv_input),
            token_positions,
            theta=self.rope_theta,
            rope_mode=self.rope_mode,
        )
        v = self.v_proj(kv_input)
        scale = 1.0 / math.sqrt(self.dim)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        return self.out_proj(torch.matmul(weights, v))


class PerceiverBlock(nn.Module):
    """One STILL perceiver block: cross-attend, then coordinate latents."""

    def __init__(
        self,
        *,
        dim: int,
        rope_theta: float,
        rope_mode: str = "default",
        active_identity_path: bool,
    ) -> None:
        super().__init__()
        self.cross_attn = LatentCrossAttention(
            dim=dim,
            rope_theta=rope_theta,
            rope_mode=rope_mode,
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
        latents = latents + self.cross_attn(
            self.cross_norm(latents),
            kv_input,
            latent_positions=latent_positions,
            token_positions=token_positions,
        )
        return latents + self.self_attn(self.self_norm(latents))


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
        beta_base: str = "log_compression",
        beta_init: float = 0.0,
        num_key_value_heads: int = 0,
        head_specific_latents: bool = False,
        rope_mode: str = "default",
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if beta_base not in {"log_compression", "zero"}:
            raise ValueError("beta_base must be 'log_compression' or 'zero'")
        if rope_mode not in ROPE_MODES:
            raise ValueError(f"rope_mode must be one of {sorted(ROPE_MODES)}")
        self.head_dim = int(head_dim)
        self.num_latents = int(num_latents)
        self.latent_dim = self.head_dim * 2
        self.rope_theta = float(rope_theta)
        self.rope_mode = rope_mode
        self.latent_dropout = float(latent_dropout)
        self.beta_base = beta_base
        self.beta_init = float(beta_init)
        self.head_specific_latents = bool(head_specific_latents)
        self.num_key_value_heads = int(num_key_value_heads)
        if self.head_specific_latents and self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive with head_specific_latents")

        latent_shape = (
            (self.num_key_value_heads, self.num_latents, self.latent_dim)
            if self.head_specific_latents
            else (self.num_latents, self.latent_dim)
        )
        self.latents = nn.Parameter(torch.zeros(*latent_shape))
        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(
                    dim=self.latent_dim,
                    rope_theta=self.rope_theta,
                    rope_mode=self.rope_mode,
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
            self.beta_head.bias.fill_(self.beta_init)

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
        if self.head_specific_latents and heads != self.num_key_value_heads:
            raise ValueError(
                f"expected {self.num_key_value_heads} KV heads for head-specific "
                f"latents, got {heads}"
            )

        dtype = keys.dtype
        module_dtype = self.key_head.weight.dtype
        token_positions = torch.arange(seq_len, device=keys.device, dtype=torch.long)
        latent_positions = evenly_spaced_positions(self.num_latents, seq_len, keys.device)

        unrotated_keys = _apply_rope_mode(
            keys,
            token_positions,
            theta=self.rope_theta,
            rope_mode=self.rope_mode,
            inverse=True,
        )
        kv_input = torch.cat([unrotated_keys, values], dim=-1)
        kv_input = kv_input.reshape(batch * heads, seq_len, self.latent_dim).to(module_dtype)
        if self.head_specific_latents:
            latents = (
                self.latents.unsqueeze(0)
                .expand(batch, heads, -1, -1)
                .reshape(batch * heads, self.num_latents, self.latent_dim)
                .to(module_dtype)
            )
        else:
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
        if self.beta_base == "log_compression":
            beta = beta + math.log(max(seq_len / self.num_latents, 1e-6))
        compact_keys = _apply_rope_mode(
            compact_keys,
            latent_positions,
            theta=self.rope_theta,
            rope_mode=self.rope_mode,
        )

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
        beta_base: str = "log_compression",
        beta_init: float = 0.0,
        layer_compactor_groups: int = 0,
        sink_tokens: int = 0,
        exact_tokens: int = 0,
        exact_strategy: str = "prefix",
        exact_beta: float = 0.0,
        num_key_value_heads: int = 0,
        head_specific_latents: bool = False,
        rope_mode: str = "default",
    ) -> None:
        super().__init__()
        if rope_mode not in ROPE_MODES:
            raise ValueError(f"rope_mode must be one of {sorted(ROPE_MODES)}")
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_latents = int(num_latents)
        self.rope_mode = rope_mode
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_specific_latents = bool(head_specific_latents)
        if self.head_specific_latents and self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive with head_specific_latents")
        self.sink_tokens = int(sink_tokens)
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        self.exact_tokens = int(exact_tokens)
        if self.exact_tokens < 0:
            raise ValueError("exact_tokens must be non-negative")
        self.exact_beta = float(exact_beta)
        if exact_strategy not in EXACT_TOKEN_STRATEGIES:
            raise ValueError(f"exact_strategy must be one of {sorted(EXACT_TOKEN_STRATEGIES)}")
        self.exact_strategy = exact_strategy
        groups = int(layer_compactor_groups or self.num_hidden_layers)
        if groups <= 0:
            raise ValueError("layer_compactor_groups must be positive or 0 for per-layer")
        if groups > self.num_hidden_layers:
            raise ValueError("layer_compactor_groups cannot exceed num_hidden_layers")
        self.layer_compactor_groups = groups
        self.beta_base = beta_base
        self.beta_init = float(beta_init)
        self.layers = nn.ModuleList(
            [
                StillLayerCompactor(
                    head_dim=head_dim,
                    num_latents=num_latents,
                    rope_theta=rope_theta,
                    num_blocks=num_blocks,
                    latent_dropout=latent_dropout,
                    beta_base=beta_base,
                    beta_init=self.beta_init,
                    num_key_value_heads=self.num_key_value_heads,
                    head_specific_latents=self.head_specific_latents,
                    rope_mode=self.rope_mode,
                )
                for _ in range(groups)
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
        beta_base: str = "log_compression",
        beta_init: float = 0.0,
        layer_compactor_groups: int = 0,
        sink_tokens: int = 0,
        exact_tokens: int = 0,
        exact_strategy: str = "prefix",
        exact_beta: float = 0.0,
        head_specific_latents: bool = False,
        rope_mode: str = "default",
    ) -> StillCompactor:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        return cls(
            num_hidden_layers=int(config.num_hidden_layers),
            head_dim=int(head_dim),
            num_latents=num_latents,
            rope_theta=rope_theta,
            num_blocks=num_blocks,
            latent_dropout=latent_dropout,
            beta_base=beta_base,
            beta_init=beta_init,
            layer_compactor_groups=layer_compactor_groups,
            sink_tokens=sink_tokens,
            exact_tokens=exact_tokens,
            exact_strategy=exact_strategy,
            exact_beta=exact_beta,
            num_key_value_heads=int(num_key_value_heads),
            head_specific_latents=head_specific_latents,
            rope_mode=rope_mode,
        )

    def _layer_group_index(self, layer_index: int) -> int:
        if self.layer_compactor_groups == self.num_hidden_layers:
            return layer_index
        return (layer_index * self.layer_compactor_groups) // self.num_hidden_layers

    @property
    def compact_tokens_per_layer(self) -> int:
        return self.sink_tokens + self.exact_tokens + self.num_latents

    def _exact_indices(
        self,
        layer_keys: torch.Tensor,
        layer_values: torch.Tensor,
        *,
        sink_count: int,
        explicit_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.exact_tokens == 0:
            return None
        seq_len = int(layer_keys.shape[-2])
        available = seq_len - sink_count
        if available <= 0:
            return None
        exact_count = min(self.exact_tokens, available)
        if explicit_indices is not None:
            indices = explicit_indices.to(device=layer_keys.device, dtype=torch.long)
            if indices.dim() == 1:
                indices = indices[(indices >= sink_count) & (indices < seq_len)]
                if indices.numel() == 0:
                    return None
                return indices.unique(sorted=True)[:exact_count]
            if indices.dim() == 3:
                indices = indices.clamp(min=sink_count, max=seq_len - 1)
                return indices[..., :exact_count].sort(dim=-1).values
            raise ValueError("explicit exact token indices must be rank 1 or 3")
        device = layer_keys.device
        if self.exact_strategy == "prefix":
            return torch.arange(
                sink_count,
                sink_count + exact_count,
                device=device,
                dtype=torch.long,
            )
        if self.exact_strategy == "even":
            if exact_count == available:
                return torch.arange(sink_count, seq_len, device=device, dtype=torch.long)
            return torch.linspace(
                sink_count,
                seq_len - 1,
                exact_count,
                device=device,
            ).round().long()
        if self.exact_strategy == "kv_norm":
            scores = layer_keys.float().square().sum(dim=-1)
            scores = scores + layer_values.float().square().sum(dim=-1)
            if sink_count:
                scores = scores.clone()
                scores[..., :sink_count] = -torch.inf
            indices = torch.topk(scores, k=exact_count, dim=-1).indices
            return indices.sort(dim=-1).values
        if self.exact_strategy == "lexical":
            return None
        raise ValueError(f"Unsupported exact_strategy: {self.exact_strategy}")

    @staticmethod
    def _gather_exact_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 1:
            gather_index = indices.view(1, 1, -1, 1).expand(
                tensor.shape[0],
                tensor.shape[1],
                -1,
                tensor.shape[-1],
            )
        elif indices.dim() == 3:
            gather_index = indices.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                tensor.shape[-1],
            )
        else:
            raise ValueError("exact token indices must be rank 1 or 3")
        return torch.gather(tensor, dim=-2, index=gather_index)

    def forward(
        self,
        past_key_values,
        *,
        metadata: dict[str, object] | None = None,
        exact_token_indices: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> CompactKVCache:
        normalized = normalize_past_key_values(past_key_values)
        if len(normalized) != self.num_hidden_layers:
            raise ValueError(
                f"expected {self.num_hidden_layers} cache layers, got {len(normalized)}"
            )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for layer_index, (layer_keys, layer_values) in enumerate(normalized):
            layer_compactor = self.layers[self._layer_group_index(layer_index)]
            layer_device = layer_keys.device
            compactor_device = next(layer_compactor.parameters()).device
            if compactor_device != layer_device:
                layer_compactor.to(layer_device)
            compact_k, compact_v, beta = layer_compactor(layer_keys, layer_values)
            sink_count = 0
            if self.sink_tokens:
                sink_count = min(self.sink_tokens, int(layer_keys.shape[-2]))
                if sink_count > 0:
                    sink_k = layer_keys[..., :sink_count, :]
                    sink_v = layer_values[..., :sink_count, :]
                    sink_beta = beta.new_zeros(*layer_keys.shape[:-2], sink_count)
                    compact_k = torch.cat([sink_k, compact_k], dim=-2)
                    compact_v = torch.cat([sink_v, compact_v], dim=-2)
                    beta = torch.cat([sink_beta, beta], dim=-1)
            explicit_indices = None
            if isinstance(exact_token_indices, list):
                explicit_indices = exact_token_indices[layer_index]
            elif exact_token_indices is not None:
                explicit_indices = exact_token_indices
            exact_indices = self._exact_indices(
                layer_keys,
                layer_values,
                sink_count=sink_count,
                explicit_indices=explicit_indices,
            )
            if exact_indices is not None:
                exact_k = self._gather_exact_tokens(layer_keys, exact_indices)
                exact_v = self._gather_exact_tokens(layer_values, exact_indices)
                exact_beta = beta.new_full(exact_k.shape[:-1], self.exact_beta)
                prefix_tokens = sink_count
                if prefix_tokens:
                    compact_k = torch.cat(
                        [
                            compact_k[..., :prefix_tokens, :],
                            exact_k,
                            compact_k[..., prefix_tokens:, :],
                        ],
                        dim=-2,
                    )
                    compact_v = torch.cat(
                        [
                            compact_v[..., :prefix_tokens, :],
                            exact_v,
                            compact_v[..., prefix_tokens:, :],
                        ],
                        dim=-2,
                    )
                    beta = torch.cat(
                        [beta[..., :prefix_tokens], exact_beta, beta[..., prefix_tokens:]],
                        dim=-1,
                    )
                else:
                    compact_k = torch.cat([exact_k, compact_k], dim=-2)
                    compact_v = torch.cat([exact_v, compact_v], dim=-2)
                    beta = torch.cat([exact_beta, beta], dim=-1)
            keys.append(compact_k)
            values.append(compact_v)
            biases.append(beta)
        output_metadata = dict(metadata or {})
        if self.sink_tokens:
            output_metadata["sink_tokens"] = self.sink_tokens
        if self.exact_tokens:
            output_metadata["exact_tokens"] = self.exact_tokens
            output_metadata["exact_strategy"] = self.exact_strategy
            output_metadata["exact_beta"] = self.exact_beta
        if self.sink_tokens or self.exact_tokens:
            output_metadata["latent_tokens"] = self.num_latents
        return CompactKVCache(
            keys=keys,
            values=values,
            biases=biases,
            metadata=output_metadata,
            detach=False,
        )
