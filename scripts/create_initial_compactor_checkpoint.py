#!/usr/bin/env python3
"""Create an initial STILL compactor checkpoint from an experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from neural_kv.models.compactor import StillCompactor
from neural_kv.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="Tensor dtype used for the saved compactor state dict.",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _legacy_compactor_fields(compactor_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_latents": int(compactor_config["num_latents"]),
        "sink_tokens": int(compactor_config.get("sink_tokens", 0)),
        "exact_tokens": int(compactor_config.get("exact_tokens", 0)),
        "exact_strategy": str(compactor_config.get("exact_strategy", "prefix")),
        "exact_beta": float(compactor_config.get("exact_beta", 0.0)),
        "num_blocks": int(compactor_config.get("num_blocks", 2)),
        "latent_dropout": float(compactor_config.get("latent_dropout", 0.0)),
        "beta_base": str(compactor_config.get("beta_base", "zero")),
        "beta_init": float(compactor_config.get("beta_init", 0.0)),
        "layer_compactor_groups": int(compactor_config.get("layer_compactor_groups", 0)),
        "head_specific_latents": bool(compactor_config.get("head_specific_latents", False)),
    }


def build_checkpoint_payload(
    model_config: Any,
    *,
    model_name: str,
    context_length: int,
    compactor_config: dict[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    compactor = StillCompactor.from_model_config(model_config, **compactor_config)
    compactor = compactor.to(dtype=dtype)
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in compactor.state_dict().items()
    }
    return {
        "step": 0,
        "model": model_name,
        "context_length": int(context_length),
        **_legacy_compactor_fields(compactor_config),
        "state_dict": state_dict,
        "metrics": {},
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    model_name = args.model or str(cfg["model"]["name"])
    context_length = int(args.context_length or cfg["model"].get("context_length", 8192))
    compactor_config = dict(cfg["model"]["compactor"])
    dtype = _dtype_from_name(args.dtype)

    model_config = AutoConfig.from_pretrained(model_name)
    payload = build_checkpoint_payload(
        model_config,
        model_name=model_name,
        context_length=context_length,
        compactor_config=compactor_config,
        dtype=dtype,
    )

    output_path = Path(args.output) if args.output else Path(cfg["output_dir"]) / "initial_step0.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in payload["state_dict"].values()
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "model": model_name,
                "context_length": context_length,
                "compact_tokens": int(payload["num_latents"])
                + int(payload["sink_tokens"])
                + int(payload["exact_tokens"]),
                "compression": context_length
                / (
                    int(payload["num_latents"])
                    + int(payload["sink_tokens"])
                    + int(payload["exact_tokens"])
                ),
                "dtype": str(dtype).removeprefix("torch."),
                "state_dict_bytes": state_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
