"""Patch Hugging Face attention modules to consume STILL beta biases."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
from collections.abc import Iterable, Iterator

import torch

_ACTIVE_BIASES: contextvars.ContextVar[list[torch.Tensor] | None] = contextvars.ContextVar(
    "neural_kv_active_biases",
    default=None,
)


@contextlib.contextmanager
def still_biases(biases: Iterable[torch.Tensor] | None) -> Iterator[None]:
    """Expose compact-cache beta biases to patched attention layers."""
    token = _ACTIVE_BIASES.set(list(biases) if biases is not None else None)
    try:
        yield
    finally:
        _ACTIVE_BIASES.reset(token)


def _get_arg(
    *,
    signature: inspect.Signature,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
):
    if name in kwargs:
        return kwargs[name], "kw", name
    names = list(signature.parameters)
    if name in names:
        index = names.index(name)
        if index < len(args):
            return args[index], "arg", index
    return None, None, None


def _set_arg(
    *,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    mode: str | None,
    key,
    value,
) -> tuple[tuple[object, ...], dict[str, object]]:
    if mode == "kw":
        new_kwargs = dict(kwargs)
        new_kwargs[key] = value
        return args, new_kwargs
    if mode == "arg":
        new_args = list(args)
        new_args[int(key)] = value
        return tuple(new_args), kwargs
    return args, kwargs


def _query_length(hidden_states: torch.Tensor | None) -> int:
    if hidden_states is None:
        return 1
    if hidden_states.dim() < 2:
        return 1
    return int(hidden_states.shape[1])


def _num_query_heads(layer_module, kv_heads: int) -> int:
    config = getattr(layer_module, "config", None)
    if config is not None and hasattr(config, "num_attention_heads"):
        return int(config.num_attention_heads)
    if hasattr(layer_module, "num_heads"):
        return int(layer_module.num_heads)
    if hasattr(layer_module, "num_attention_heads"):
        return int(layer_module.num_attention_heads)
    groups = int(getattr(layer_module, "num_key_value_groups", 1))
    return kv_heads * groups


def merge_still_bias(
    *,
    layer_module,
    attention_mask: torch.Tensor | None,
    hidden_states: torch.Tensor | None,
    layer_biases: list[torch.Tensor] | None,
) -> torch.Tensor | None:
    """Merge one layer's beta vector into an additive attention mask."""
    if layer_biases is None:
        return attention_mask
    layer_idx = getattr(layer_module, "layer_idx", None)
    if layer_idx is None or int(layer_idx) >= len(layer_biases):
        return attention_mask

    beta = layer_biases[int(layer_idx)]
    # beta: [batch, kv_heads, compact_tokens]
    if beta.dim() != 3:
        raise ValueError("STILL beta tensors must be shaped [batch, kv_heads, tokens]")
    batch, kv_heads, compact_tokens = beta.shape
    query_len = _query_length(hidden_states)
    num_heads = _num_query_heads(layer_module, kv_heads)
    if num_heads % kv_heads != 0:
        raise ValueError(f"Cannot repeat {kv_heads} KV-head beta values to {num_heads} query heads")

    beta = beta.to(dtype=(hidden_states.dtype if hidden_states is not None else torch.float32))
    beta = beta.repeat_interleave(num_heads // kv_heads, dim=1)
    beta = beta.unsqueeze(-2).expand(batch, num_heads, query_len, compact_tokens)

    if attention_mask is None:
        return beta

    mask = attention_mask
    if mask.dtype == torch.bool:
        additive = torch.zeros(mask.shape, device=mask.device, dtype=beta.dtype)
        additive = additive.masked_fill(~mask, torch.finfo(beta.dtype).min)
    else:
        additive = mask.to(dtype=beta.dtype)

    if additive.dim() == 2:
        additive = additive[:, None, None, :]
    elif additive.dim() == 3:
        additive = additive[:, None, :, :]
    elif additive.dim() != 4:
        raise ValueError(f"Unsupported attention mask rank for STILL bias: {additive.dim()}")

    key_len = int(additive.shape[-1])
    if key_len < compact_tokens:
        raise ValueError(
            f"Attention mask key length {key_len} is shorter than compact "
            f"beta length {compact_tokens}"
        )
    if additive.shape[-2] == 1 and query_len != 1:
        additive = additive.expand(
            additive.shape[0],
            additive.shape[1],
            query_len,
            additive.shape[-1],
        )
    if additive.shape[1] == 1 and num_heads != 1:
        additive = additive.expand(
            additive.shape[0],
            num_heads,
            additive.shape[-2],
            additive.shape[-1],
        )

    padded_beta = torch.zeros_like(additive)
    padded_beta[..., :compact_tokens] = beta.to(device=additive.device)
    return additive + padded_beta


def enable_still_attention_bias(model) -> int:
    """Patch attention modules that expose ``layer_idx`` so beta affects logits.

    The patch is intentionally narrow: it only modifies modules with a
    ``layer_idx`` attribute and a forward argument named ``attention_mask``.
    Calls outside ``with still_biases(...)`` are unchanged.
    """
    patched = 0
    for module in model.modules():
        if getattr(module, "_neural_kv_bias_patched", False):
            continue
        if not hasattr(module, "layer_idx"):
            continue
        original_forward = module.forward
        signature = inspect.signature(original_forward)
        if "attention_mask" not in signature.parameters:
            continue

        def wrapped_forward(
            *args,
            __module=module,
            __original=original_forward,
            __signature=signature,
            **kwargs,
        ):
            active = _ACTIVE_BIASES.get()
            if active is None:
                return __original(*args, **kwargs)
            attention_mask, mode, key = _get_arg(
                signature=__signature,
                args=args,
                kwargs=kwargs,
                name="attention_mask",
            )
            hidden_states, _, _ = _get_arg(
                signature=__signature,
                args=args,
                kwargs=kwargs,
                name="hidden_states",
            )
            merged = merge_still_bias(
                layer_module=__module,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                layer_biases=active,
            )
            new_args, new_kwargs = _set_arg(
                args=args,
                kwargs=kwargs,
                mode=mode,
                key=key,
                value=merged,
            )
            return __original(*new_args, **new_kwargs)

        module.forward = wrapped_forward
        module._neural_kv_bias_patched = True
        patched += 1
    return patched
