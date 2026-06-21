"""Initialization helpers for legacy checkpoint parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neural_kv.models.checkpointing import CompactorCheckpoint, parse_compactor_checkpoint
from neural_kv.models.compactor import StillCompactor


def _normalized_compactor_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "sink_tokens": 0,
        "exact_tokens": 0,
        "exact_strategy": "prefix",
        "num_blocks": 2,
        "latent_dropout": 0.0,
        "beta_base": "zero",
        "layer_compactor_groups": 0,
        "head_specific_latents": False,
    }
    merged = {**defaults, **config}
    return {
        "num_latents": int(merged["num_latents"]),
        "sink_tokens": int(merged["sink_tokens"]),
        "exact_tokens": int(merged["exact_tokens"]),
        "exact_strategy": str(merged["exact_strategy"]),
        "num_blocks": int(merged["num_blocks"]),
        "latent_dropout": float(merged["latent_dropout"]),
        "beta_base": str(merged["beta_base"]),
        "layer_compactor_groups": int(merged["layer_compactor_groups"]),
        "head_specific_latents": bool(merged["head_specific_latents"]),
    }


def validate_init_checkpoint(
    spec: CompactorCheckpoint,
    *,
    model_name: str,
    context_length: int,
    compactor_config: dict[str, Any],
) -> None:
    """Validate a checkpoint against the requested run configuration."""
    if spec.model_name != model_name:
        raise ValueError(f"init checkpoint model {spec.model_name!r} does not match {model_name!r}")
    if int(spec.context_length) != int(context_length):
        raise ValueError("init checkpoint context_length does not match config")
    expected = _normalized_compactor_config(compactor_config)
    actual = _normalized_compactor_config(spec.compactor)
    if actual != expected:
        raise ValueError(f"init checkpoint compactor config mismatch: {actual!r} != {expected!r}")


def load_initial_compactor_state(
    *,
    compactor: StillCompactor,
    checkpoint_path: str | Path,
    model_name: str,
    context_length: int,
    compactor_config: dict[str, Any],
) -> int:
    """Load a legacy or Lightning compactor state and return its global step."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = parse_compactor_checkpoint(checkpoint)
    validate_init_checkpoint(
        spec,
        model_name=model_name,
        context_length=context_length,
        compactor_config=compactor_config,
    )
    compactor.load_state_dict(spec.state_dict)
    return int(checkpoint.get("step", checkpoint.get("global_step", 0)))
