#!/usr/bin/env python3
"""Export a legacy or Lightning neural KV checkpoint to an HF-style directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from neural_kv.models.checkpointing import parse_compactor_checkpoint
from neural_kv.models.hf import NeuralKVConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = parse_compactor_checkpoint(checkpoint, base_model=args.base_model)
    config = NeuralKVConfig(
        base_model_name_or_path=spec.model_name,
        context_length=spec.context_length,
        compactor=spec.compactor,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    save_file(
        {key: value.detach().cpu() for key, value in spec.state_dict.items()},
        output_dir / "model.safetensors",
    )
    print(f"exported neural KV checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
