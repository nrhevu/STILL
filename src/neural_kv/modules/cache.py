"""Compact KV-cache containers and accounting helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch


def normalize_past_key_values(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize Hugging Face cache objects to ``[(keys, values), ...]``."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in past_key_values]


@dataclass(frozen=True)
class KVShape:
    """Minimal cache shape metadata."""

    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @classmethod
    def from_model_config(cls, config, *, dtype_bytes: int = 2) -> KVShape:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        return cls(
            num_hidden_layers=int(config.num_hidden_layers),
            num_key_value_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype_bytes=dtype_bytes,
        )

    def bytes_for_tokens(self, tokens: int) -> int:
        return int(
            tokens
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * 2
            * self.dtype_bytes
        )


class CompactKVCache:
    """Serializable compact cache with per-layer keys, values, and beta biases."""

    def __init__(
        self,
        *,
        keys: Iterable[torch.Tensor],
        values: Iterable[torch.Tensor],
        biases: Iterable[torch.Tensor] | None = None,
        metadata: dict[str, object] | None = None,
        detach: bool = True,
    ) -> None:
        self.keys = [tensor.detach().clone() if detach else tensor for tensor in keys]
        self.values = [tensor.detach().clone() if detach else tensor for tensor in values]
        if len(self.keys) != len(self.values):
            raise ValueError("keys and values must have the same number of layers")
        if not self.keys:
            raise ValueError("CompactKVCache needs at least one layer")
        if biases is None:
            self.biases = [torch.zeros_like(key[..., 0]) for key in self.keys]
        else:
            self.biases = [tensor.detach().clone() if detach else tensor for tensor in biases]
        if len(self.biases) != len(self.keys):
            raise ValueError("biases must match the number of layers")
        self.metadata = dict(metadata or {})

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def num_tokens(self) -> int:
        return int(self.keys[0].shape[-2])

    def to(self, device: str | torch.device) -> CompactKVCache:
        self.keys = [tensor.to(device) for tensor in self.keys]
        self.values = [tensor.to(device) for tensor in self.values]
        self.biases = [tensor.to(device) for tensor in self.biases]
        return self

    def as_legacy_past_key_values(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((key, value) for key, value in zip(self.keys, self.values, strict=True))

    def as_dynamic_cache(self):
        """Return a Transformers ``DynamicCache`` for current HF model forwards."""
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(self.as_legacy_past_key_values())

    def canonical_kv_bytes(self) -> int:
        total = 0
        for key, value in zip(self.keys, self.values, strict=True):
            total += key.numel() * key.element_size()
            total += value.numel() * value.element_size()
        return int(total)

    def compression_ratio_vs(self, source_tokens: int) -> float:
        if self.num_tokens == 0:
            return float("inf")
        return float(source_tokens) / float(self.num_tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": [tensor.detach().cpu() for tensor in self.keys],
                "values": [tensor.detach().cpu() for tensor in self.values],
                "biases": [tensor.detach().cpu() for tensor in self.biases],
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> CompactKVCache:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        cache = cls(
            keys=payload["keys"],
            values=payload["values"],
            biases=payload.get("biases"),
            metadata=payload.get("metadata"),
        )
        if device is not None:
            cache.to(device)
        return cache


def prune_cache_by_beta(cache: CompactKVCache, *, keep_fraction: float) -> CompactKVCache:
    """Adaptively keep high-beta compact slots per layer/head.

    This is a practical hook for the adaptive-budget recommendation in
    ``research.md``. It does not retrain the compactor; it evaluates whether the
    learned beta scores can support a smaller downstream cache.
    """
    if not (0 < keep_fraction <= 1):
        raise ValueError("keep_fraction must be in (0, 1]")
    new_keys: list[torch.Tensor] = []
    new_values: list[torch.Tensor] = []
    new_biases: list[torch.Tensor] = []
    for key, value, beta in zip(cache.keys, cache.values, cache.biases, strict=True):
        keep = max(1, int(round(key.shape[-2] * keep_fraction)))
        score = beta.float()
        indices = torch.topk(score, k=keep, dim=-1).indices.sort(dim=-1).values
        gather_index = indices.unsqueeze(-1).expand(*indices.shape, key.shape[-1])
        new_keys.append(torch.gather(key, dim=-2, index=gather_index))
        new_values.append(torch.gather(value, dim=-2, index=gather_index))
        new_biases.append(torch.gather(beta, dim=-1, index=indices))
    metadata = dict(cache.metadata)
    metadata["beta_keep_fraction"] = keep_fraction
    return CompactKVCache(keys=new_keys, values=new_values, biases=new_biases, metadata=metadata)
