"""Vision-language compactor evaluation helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from neural_kv.data.vlm import (
    VLMExample,
    format_vlm_source_prompt,
    resize_image_max_side,
    score_vlm_prediction,
    vlm_answer_prompt,
)
from neural_kv.models.compactor import StillCompactor
from neural_kv.modules.attention_bias import still_biases
from neural_kv.modules.cache import normalize_past_key_values

QWEN_VL_PATCH_SIZE = 28


@dataclass(frozen=True)
class EncodedVLMExample:
    """Processor outputs for a source multimodal prefix plus text continuation prompt."""

    source_inputs: dict[str, Any]
    prompt_ids: torch.Tensor
    source_text: str
    prompt_text: str
    source_tokens: int
    visual_tokens: int | None
    original_size: tuple[int, int] | None
    resized_size: tuple[int, int] | None


def image_token_budget_to_pixels(tokens: int | None) -> int | None:
    if tokens is None or tokens <= 0:
        return None
    return int(tokens) * QWEN_VL_PATCH_SIZE * QWEN_VL_PATCH_SIZE


def model_input_device(model, fallback: str = "cpu") -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def load_vlm_processor(
    model_name: str,
    *,
    image_token_budget: int | None = None,
    trust_remote_code: bool = False,
):
    """Load a processor, applying Qwen-VL pixel bounds for a requested token budget."""
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    pixels = image_token_budget_to_pixels(image_token_budget)
    if pixels is not None:
        kwargs["min_pixels"] = pixels
        kwargs["max_pixels"] = pixels
    return AutoProcessor.from_pretrained(model_name, **kwargs)


def _vlm_model_class():
    import transformers

    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("This transformers version does not expose a supported VLM model class")


def load_vlm_model(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    device_map: str | None = None,
    trust_remote_code: bool = False,
):
    model_cls = _vlm_model_class()
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype if device.startswith("cuda") or device_map else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    if not device_map:
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _messages_for_source(image: Any, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
        }
    ]


def _apply_vlm_template(processor, *, image: Any, text: str) -> str:
    messages = _messages_for_source(image, text)
    template = getattr(processor, "apply_chat_template", None)
    if template is None:
        return "<image>\n" + text
    try:
        return template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return template(messages, tokenize=False)


def _tokenize_prompt(processor, text: str, *, device: torch.device) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device=device)


def _to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    output = dict(inputs)
    for key, value in list(output.items()):
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device=device)
    return output


def _visual_tokens_from_inputs(inputs: Mapping[str, Any]) -> int | None:
    grid = inputs.get("image_grid_thw")
    if isinstance(grid, torch.Tensor) and grid.numel() > 0:
        return int(grid.long().prod(dim=-1).sum().item())
    pixel_values = inputs.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() >= 2:
        return int(pixel_values.shape[0])
    return None


def encode_vlm_example(
    processor,
    example: VLMExample,
    *,
    device: torch.device,
    resolution: int | None = None,
    no_think: bool = True,
) -> EncodedVLMExample:
    """Encode one image/question row as a cacheable multimodal prefix."""
    original_size = getattr(example.image, "size", None)
    image = resize_image_max_side(example.image, resolution)
    resized_size = getattr(image, "size", None)
    source_text = format_vlm_source_prompt(example, no_think=no_think)
    rendered_source = _apply_vlm_template(processor, image=image, text=source_text)
    processor_inputs = processor(
        text=[rendered_source],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    source_inputs = _to_device(processor_inputs, device)
    prompt_text = vlm_answer_prompt(example)
    prompt_ids = _tokenize_prompt(processor, prompt_text, device=device)
    source_ids = source_inputs["input_ids"]
    return EncodedVLMExample(
        source_inputs=source_inputs,
        prompt_ids=prompt_ids,
        source_text=source_text,
        prompt_text=prompt_text,
        source_tokens=int(source_ids.shape[-1]),
        visual_tokens=_visual_tokens_from_inputs(source_inputs),
        original_size=tuple(original_size) if original_size is not None else None,
        resized_size=tuple(resized_size) if resized_size is not None else None,
    )


def _position_ids(start: int, length: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


def _attention_mask(prefix_length: int, input_length: int, *, device: torch.device) -> torch.Tensor:
    return torch.ones(1, prefix_length + input_length, device=device, dtype=torch.long)


def _fresh_dynamic_cache(past_key_values):
    if past_key_values is None:
        return None
    from transformers.cache_utils import DynamicCache

    legacy = normalize_past_key_values(past_key_values)
    return DynamicCache.from_legacy_cache(tuple((key, value) for key, value in legacy))


def _forward_with_optional_biases(model, biases, **kwargs):
    if biases is None:
        return model(**kwargs)
    with still_biases(biases):
        return model(**kwargs)


def _forward_text_with_cache(
    model,
    *,
    biases: list[torch.Tensor] | None,
    input_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    use_cache: bool,
):
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
    }
    try:
        return _forward_with_optional_biases(model, biases, **kwargs)
    except (RuntimeError, TypeError, ValueError) as exc:
        message = str(exc).lower()
        if "position" not in message and "rope" not in message and "mrope" not in message:
            raise
        kwargs.pop("position_ids")
        return _forward_with_optional_biases(model, biases, **kwargs)


@torch.no_grad()
def greedy_generate_from_cache(
    *,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    past_key_values,
    cache_tokens: int,
    source_position_start: int,
    max_new_tokens: int,
    biases: list[torch.Tensor] | None = None,
) -> str:
    """Generate from a text prompt appended to a cached multimodal prefix."""
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if biases is not None:
        generated: list[int] = []
        for _ in range(max_new_tokens):
            if generated:
                generated_ids = torch.tensor(
                    [generated],
                    device=prompt_ids.device,
                    dtype=prompt_ids.dtype,
                )
                model_inputs = torch.cat([prompt_ids, generated_ids], dim=-1)
            else:
                model_inputs = prompt_ids
            outputs = _forward_text_with_cache(
                model,
                biases=biases,
                input_ids=model_inputs,
                past_key_values=_fresh_dynamic_cache(past_key_values),
                attention_mask=_attention_mask(
                    cache_tokens,
                    int(model_inputs.shape[-1]),
                    device=prompt_ids.device,
                ),
                position_ids=_position_ids(
                    source_position_start,
                    int(model_inputs.shape[-1]),
                    device=prompt_ids.device,
                ),
                use_cache=False,
            )
            next_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            if eos_id is not None and next_id == int(eos_id):
                break
            generated.append(next_id)
        return tokenizer.decode(generated, skip_special_tokens=True)

    outputs = _forward_text_with_cache(
        model,
        biases=biases,
        input_ids=prompt_ids,
        past_key_values=past_key_values,
        attention_mask=_attention_mask(cache_tokens, prompt_len, device=prompt_ids.device),
        position_ids=_position_ids(source_position_start, prompt_len, device=prompt_ids.device),
        use_cache=True,
    )
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    generated: list[int] = []

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if eos_id is not None and token_id == int(eos_id):
            break
        generated.append(token_id)
        past_tokens = cache_tokens + prompt_len + len(generated) - 1
        position = source_position_start + prompt_len + len(generated) - 1
        outputs = _forward_text_with_cache(
            model,
            biases=biases,
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=_attention_mask(past_tokens, 1, device=prompt_ids.device),
            position_ids=_position_ids(position, 1, device=prompt_ids.device),
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def evaluate_vlm_example(
    *,
    model,
    processor,
    example: VLMExample,
    compactor: StillCompactor,
    device: torch.device,
    resolution: int | None,
    image_token_budget: int | None,
    max_new_tokens: int,
    no_think: bool = True,
) -> dict[str, Any]:
    """Evaluate one example with full and compacted source caches."""
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = encode_vlm_example(
        processor,
        example,
        device=device,
        resolution=resolution,
        no_think=no_think,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    source_outputs = model(**encoded.source_inputs, use_cache=True)
    full_text = greedy_generate_from_cache(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=encoded.prompt_ids,
        past_key_values=source_outputs.past_key_values,
        cache_tokens=encoded.source_tokens,
        source_position_start=encoded.source_tokens,
        max_new_tokens=max_new_tokens,
        biases=None,
    )
    full_seconds = time.perf_counter() - start

    compact_start = time.perf_counter()
    compact_cache = compactor(
        source_outputs.past_key_values,
        metadata={
            "source_tokens": encoded.source_tokens,
            "visual_tokens": encoded.visual_tokens,
            "resolution": resolution,
            "image_token_budget": image_token_budget,
        },
    )
    compact_text = greedy_generate_from_cache(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=encoded.prompt_ids,
        past_key_values=compact_cache.as_dynamic_cache(),
        cache_tokens=compact_cache.num_tokens,
        source_position_start=encoded.source_tokens,
        max_new_tokens=max_new_tokens,
        biases=compact_cache.biases,
    )
    compact_seconds = time.perf_counter() - compact_start
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    full_prediction, full_correct = score_vlm_prediction(example, full_text)
    compact_prediction, compact_correct = score_vlm_prediction(example, compact_text)
    full_reference_valid = full_prediction is not None
    compact_matches_full = full_reference_valid and compact_prediction == full_prediction
    return {
        "id": example.id,
        "dataset": example.dataset,
        "task": example.task,
        "split": example.split,
        "subject": example.subject,
        "resolution": resolution,
        "image_token_budget": image_token_budget,
        "source_tokens": encoded.source_tokens,
        "visual_tokens": encoded.visual_tokens,
        "compact_tokens": compact_cache.num_tokens,
        "compression": float(encoded.source_tokens / max(compact_cache.num_tokens, 1)),
        "original_size": encoded.original_size,
        "resized_size": encoded.resized_size,
        "answers": example.answers,
        "answer_letter": example.answer_letter,
        "full_text": full_text,
        "full_prediction": full_prediction,
        "full_correct": bool(full_correct),
        "compact_text": compact_text,
        "compact_prediction": compact_prediction,
        "compact_correct": bool(compact_correct),
        "full_reference_valid": bool(full_reference_valid),
        "compact_matches_full": bool(compact_matches_full),
        "full_seconds": full_seconds,
        "compact_seconds": compact_seconds,
        "peak_memory_bytes": int(peak_bytes),
    }


def compact_vs_full_accuracy(
    *,
    compact_accuracy: float,
    full_accuracy: float,
) -> float | None:
    """Return compact gold accuracy as a fraction of full-cache gold accuracy."""
    if full_accuracy <= 0:
        return None
    return compact_accuracy / full_accuracy


def _accuracy_pair(items: list[dict[str, Any]]) -> tuple[float, float]:
    count = len(items)
    full_accuracy = sum(bool(item["full_correct"]) for item in items) / count
    compact_accuracy = sum(bool(item["compact_correct"]) for item in items) / count
    return full_accuracy, compact_accuracy


def _compact_full_agreement(items: list[dict[str, Any]]) -> tuple[int, float | None]:
    valid = [item for item in items if bool(item.get("full_reference_valid", True))]
    if not valid:
        return 0, None
    matches = sum(bool(item.get("compact_matches_full")) for item in valid)
    return len(valid), matches / len(valid)


def summarize_vlm_results(
    rows: list[dict[str, Any]],
    *,
    target_compact_vs_full_accuracy: float = 0.95,
) -> dict[str, Any]:
    """Aggregate VLM detail rows by task/resolution/token budget."""
    groups: dict[tuple[str, int | None, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["task"]),
            row.get("resolution"),
            row.get("image_token_budget"),
        )
        groups.setdefault(key, []).append(row)

    by_group: list[dict[str, Any]] = []
    failing_groups: list[str] = []
    for (task, resolution, budget), items in sorted(groups.items(), key=lambda item: str(item[0])):
        count = len(items)
        full_accuracy, compact_accuracy = _accuracy_pair(items)
        relative_accuracy = compact_vs_full_accuracy(
            compact_accuracy=compact_accuracy,
            full_accuracy=full_accuracy,
        )
        reference_count, agreement = _compact_full_agreement(items)
        target_passed = (
            agreement is not None
            and agreement >= target_compact_vs_full_accuracy
        )
        group_name = f"{task}:resolution={resolution}:image_token_budget={budget}"
        if not target_passed:
            failing_groups.append(group_name)
        by_group.append(
            {
                "task": task,
                "resolution": resolution,
                "image_token_budget": budget,
                "count": count,
                "full_accuracy": full_accuracy,
                "compact_accuracy": compact_accuracy,
                "compact_vs_full_accuracy": relative_accuracy,
                "compact_full_agreement": agreement,
                "full_reference_count": reference_count,
                "target_passed": target_passed,
                "avg_source_tokens": sum(float(item["source_tokens"]) for item in items) / count,
                "avg_visual_tokens": sum(float(item.get("visual_tokens") or 0) for item in items)
                / count,
                "avg_compact_tokens": sum(float(item["compact_tokens"]) for item in items) / count,
                "avg_compression": sum(float(item["compression"]) for item in items) / count,
                "avg_full_seconds": sum(float(item["full_seconds"]) for item in items) / count,
                "avg_compact_seconds": sum(float(item["compact_seconds"]) for item in items)
                / count,
            }
        )
    total = len(rows)
    if total:
        full_accuracy, compact_accuracy = _accuracy_pair(rows)
    else:
        full_accuracy, compact_accuracy = 0.0, 0.0
    relative_accuracy = compact_vs_full_accuracy(
        compact_accuracy=compact_accuracy,
        full_accuracy=full_accuracy,
    )
    reference_count, agreement = _compact_full_agreement(rows)
    aggregate_passed = (
        agreement is not None
        and agreement >= target_compact_vs_full_accuracy
    )
    return {
        "count": total,
        "full_accuracy": full_accuracy,
        "compact_accuracy": compact_accuracy,
        "compact_vs_full_accuracy": relative_accuracy,
        "compact_full_agreement": agreement,
        "full_reference_count": reference_count,
        "target_compact_vs_full_accuracy": target_compact_vs_full_accuracy,
        "target_passed": aggregate_passed and not failing_groups,
        "aggregate_target_passed": aggregate_passed,
        "failing_groups": failing_groups,
        "groups": by_group,
    }
