"""Checkpoint metadata helpers for legacy and Lightning formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CompactorCheckpoint:
    model_name: str
    context_length: int
    compactor: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def legacy_compactor_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_latents": int(checkpoint["num_latents"]),
        "sink_tokens": int(checkpoint.get("sink_tokens", 0)),
        "exact_tokens": int(checkpoint.get("exact_tokens", 0)),
        "exact_strategy": str(checkpoint.get("exact_strategy", "prefix")),
        "num_blocks": int(checkpoint.get("num_blocks", 2)),
        "latent_dropout": float(checkpoint.get("latent_dropout", 0.0)),
        "beta_base": str(checkpoint.get("beta_base", "zero")),
        "layer_compactor_groups": int(checkpoint.get("layer_compactor_groups", 0)),
        "head_specific_latents": bool(checkpoint.get("head_specific_latents", False)),
    }


def parse_compactor_checkpoint(
    checkpoint: dict[str, Any],
    *,
    base_model: str | None = None,
) -> CompactorCheckpoint:
    """Parse either a legacy ``.pt`` checkpoint or a Lightning ``.ckpt``."""
    if "neural_kv" in checkpoint:
        metadata = checkpoint["neural_kv"]
        state_dict = {
            key.removeprefix("compactor."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("compactor.")
        }
        if not state_dict:
            raise ValueError("Lightning checkpoint contains no compactor.* state dict entries")
        return CompactorCheckpoint(
            model_name=base_model or str(metadata["model"]),
            context_length=int(metadata["context_length"]),
            compactor=dict(metadata["compactor"]),
            state_dict=state_dict,
        )

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a compactor state_dict")
    return CompactorCheckpoint(
        model_name=base_model or str(checkpoint["model"]),
        context_length=int(checkpoint["context_length"]),
        compactor=legacy_compactor_config(checkpoint),
        state_dict=state_dict,
    )
