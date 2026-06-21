"""Hugging Face-style save/load wrapper for neural KV compactors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neural_kv.models.compactor import StillCompactor
from neural_kv.modules.attention_bias import enable_still_attention_bias, still_biases
from neural_kv.training.distillation import (
    dtype_from_name,
    load_model_and_tokenizer,
    resolve_device,
)


@dataclass(frozen=True)
class NeuralKVConfig:
    """Serializable metadata needed to reconstruct a neural KV compactor."""

    base_model_name_or_path: str
    context_length: int
    compactor: dict[str, Any]
    neural_kv_version: str = "1"
    model_type: str = "neural_kv"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        (output_path / "neural_kv_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        hf_payload = {
            "model_type": self.model_type,
            "architectures": ["NeuralKVForCausalLM"],
            "base_model_name_or_path": self.base_model_name_or_path,
            "neural_kv_config": payload,
        }
        (output_path / "config.json").write_text(
            json.dumps(hf_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> NeuralKVConfig:
        path = Path(path)
        config_path = path / "neural_kv_config.json"
        if not config_path.exists():
            config_path = path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "neural_kv_config" in payload:
            payload = payload["neural_kv_config"]
        return cls(**payload)


class NeuralKVForCausalLM(nn.Module):
    """Frozen HF causal LM plus a trainable or exported neural KV compactor."""

    def __init__(self, *, base_model, tokenizer, compactor: StillCompactor, config: NeuralKVConfig):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.compactor = compactor
        self.neural_kv_config = config
        self.patched_attention_layers = enable_still_attention_bias(self.base_model)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> NeuralKVForCausalLM:
        from safetensors.torch import load_file

        path = Path(path)
        config = NeuralKVConfig.from_pretrained(path)
        resolved_device = resolve_device(device)
        base_model, tokenizer = load_model_and_tokenizer(
            config.base_model_name_or_path,
            device=resolved_device,
            dtype=dtype_from_name(dtype),
        )
        compactor = StillCompactor.from_model_config(base_model.config, **config.compactor)
        state_dict = load_file(path / "model.safetensors", device="cpu")
        compactor.load_state_dict(state_dict)
        compactor.to(resolved_device)
        compactor.eval()
        return cls(base_model=base_model, tokenizer=tokenizer, compactor=compactor, config=config)

    def save_pretrained(self, output_dir: str | Path) -> None:
        from safetensors.torch import save_file

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.neural_kv_config.save_pretrained(output_path)
        save_file(
            {key: value.detach().cpu() for key, value in self.compactor.state_dict().items()},
            output_path / "model.safetensors",
        )

    @torch.no_grad()
    def compact_context(self, input_ids: torch.Tensor):
        outputs = self.base_model(input_ids=input_ids, use_cache=True)
        return self.compactor(
            outputs.past_key_values,
            metadata={"source_tokens": int(input_ids.shape[-1])},
        )

    def forward_with_compact_cache(self, *, input_ids: torch.Tensor, compact_cache, **kwargs):
        with still_biases(compact_cache.biases):
            return self.base_model(
                input_ids=input_ids,
                past_key_values=compact_cache.as_dynamic_cache(),
                **kwargs,
            )
