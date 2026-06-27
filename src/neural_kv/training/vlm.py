"""Training and scoring helpers for VLM KV-cache compaction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from neural_kv.data.vlm import (
    choice_labels,
    format_vlm_mcq_prompt,
    resolve_image_paths,
    vlm_answer_letter,
)
from neural_kv.models.compactor import StillCompactor
from neural_kv.modules.attention_bias import enable_still_attention_bias, still_biases
from neural_kv.training.distillation import dtype_from_name, resolve_device
from neural_kv.training.distillation import _fresh_dynamic_cache as fresh_dynamic_cache
from neural_kv.utils.hf_cache import configure_hf_cache

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - exercised only without train extra
    pl = None  # type: ignore[assignment]


configure_hf_cache()

VLM_ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer|option|choice|correct\s+answer)"
    r"\s*(?:is|:|-)?\s*(?:option\s*)?[\(\[]?([A-Z])",
    re.IGNORECASE,
)
VLM_TAIL_LETTER_PATTERN = re.compile(r"(?:^|[^A-Za-z])([A-Z])(?:[^A-Za-z]|$)")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return str(device)


def text_config_for_compactor(model_config):
    return getattr(model_config, "text_config", model_config)


def load_vlm_model_and_processor(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    device_map: str | None = None,
):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM

        model_cls = AutoModelForMultimodalLM
    except ImportError:  # pragma: no cover - depends on transformers version
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype if device.startswith("cuda") or device_map else torch.float32,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    if not device_map:
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def configure_vlm_processor_image_budget(
    processor,
    *,
    min_tokens: int = 0,
    max_tokens: int = 0,
    min_pixels: int = 0,
    max_pixels: int = 0,
) -> dict[str, int]:
    """Apply a Qwen3-VL image token budget and return the effective pixel budget."""
    if min_tokens <= 0 and max_tokens <= 0 and min_pixels <= 0 and max_pixels <= 0:
        return {}
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("VLM processor does not expose an image_processor")
    size = dict(getattr(image_processor, "size", {}) or {})
    if max_pixels > 0:
        size["longest_edge"] = int(max_pixels)
    elif max_tokens > 0:
        size["longest_edge"] = int(max_tokens) * 32 * 32
    if min_pixels > 0:
        size["shortest_edge"] = int(min_pixels)
    elif min_tokens > 0:
        size["shortest_edge"] = int(min_tokens) * 32 * 32
    image_processor.size = size
    return {key: int(value) for key, value in size.items()}


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def build_vlm_messages(
    row: dict[str, Any],
    *,
    base_dir: str | Path,
    prompt_style: str = "compact",
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in resolve_image_paths(row, base_dir=base_dir):
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": format_vlm_mcq_prompt(row, prompt_style=prompt_style)})
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": content})
    return messages


def encode_vlm_prompt(
    *,
    processor,
    row: dict[str, Any],
    base_dir: str | Path,
    device: str,
    prompt_style: str = "compact",
    system_prompt: str = "",
) -> dict[str, Any]:
    messages = build_vlm_messages(
        row,
        base_dir=base_dir,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return move_batch_to_device(dict(inputs), device)


def split_source_and_tail(inputs: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor, int]:
    input_ids = inputs["input_ids"]
    if int(input_ids.shape[-1]) < 2:
        raise ValueError("VLM prompt must contain at least two tokens")
    source_len = int(input_ids.shape[-1]) - 1
    source_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
            source_inputs[key] = value[:, :source_len]
        else:
            source_inputs[key] = value
    return source_inputs, input_ids[:, source_len:], source_len


def continuation_attention_mask(
    *,
    cache_tokens: int,
    input_tokens: int,
    device: str,
) -> torch.Tensor:
    return torch.ones(1, cache_tokens + input_tokens, device=device, dtype=torch.long)


def continuation_position_ids(
    *,
    source_tokens: int,
    input_tokens: int,
    device: str,
) -> torch.Tensor:
    return torch.arange(
        source_tokens,
        source_tokens + input_tokens,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)


def continuation_mrope_position_ids(
    *,
    model,
    source_tokens: int,
    input_tokens: int,
    device: str,
) -> torch.Tensor:
    rope_deltas = getattr(getattr(model, "model", None), "rope_deltas", None)
    if rope_deltas is None:
        return continuation_position_ids(
            source_tokens=source_tokens,
            input_tokens=input_tokens,
            device=device,
        )
    rope_deltas = rope_deltas.to(device=device)
    batch_size = int(rope_deltas.shape[0])
    base = torch.arange(
        source_tokens,
        source_tokens + input_tokens,
        device=device,
        dtype=torch.long,
    ).view(1, 1, -1)
    base = base.expand(3, batch_size, -1)
    return base + rope_deltas.view(1, batch_size, 1)


def _tokenizer_from_processor(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("VLM processor does not expose a tokenizer")
    return tokenizer


def letter_token_ids(processor, labels: list[str], *, device: str) -> list[int]:
    tokenizer = _tokenizer_from_processor(processor)
    ids: list[int] = []
    for label in labels:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if not token_ids:
            raise ValueError(f"Tokenizer produced no ids for label {label!r}")
        ids.append(int(token_ids[0]))
    return ids


def choice_logits(logits: torch.Tensor, label_ids: list[int]) -> torch.Tensor:
    return logits[:, -1, label_ids].float()


def predict_letter_from_choice_logits(row: dict[str, Any], logits: torch.Tensor) -> str:
    labels = choice_labels(len(row["choices"]))
    index = int(torch.argmax(logits, dim=-1).item())
    return labels[index]


def extract_vlm_answer_letter(text: str, row: dict[str, Any]) -> str:
    labels = set(choice_labels(len(row["choices"])))
    matches = [
        match.group(1).upper()
        for match in VLM_ANSWER_PATTERN.finditer(text)
        if match.group(1).upper() in labels
    ]
    if matches:
        return matches[-1]

    tail = text[-800:]
    choice_hits: list[str] = []
    for label, choice in zip(choice_labels(len(row["choices"])), row["choices"], strict=True):
        choice_text = str(choice).strip()
        if not choice_text:
            continue
        if re.search(
            rf"(?:answer|option|choice|correct).{0,80}{re.escape(choice_text)}",
            tail,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            choice_hits.append(label)
    if len(choice_hits) == 1:
        return choice_hits[0]

    tail_matches = [
        match.group(1).upper()
        for match in VLM_TAIL_LETTER_PATTERN.finditer(tail)
        if match.group(1).upper() in labels
    ]
    return tail_matches[-1] if tail_matches else ""


def letter_kl_ce_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_index: int,
    kl_weight: float,
    ce_weight: float,
) -> torch.Tensor:
    loss = student_logits.new_tensor(0.0, dtype=torch.float32)
    if kl_weight > 0:
        teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
        student_log_prob = torch.log_softmax(student_logits.float(), dim=-1)
        loss = loss + kl_weight * F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    if ce_weight > 0:
        target = torch.tensor([target_index], device=student_logits.device, dtype=torch.long)
        loss = loss + ce_weight * F.cross_entropy(student_logits.float(), target)
    return loss


def _model_forward_tail(
    *,
    model,
    tail_ids: torch.Tensor,
    past_key_values,
    cache_tokens: int,
    source_tokens: int,
    device: str,
    include_current_in_mask: bool = True,
):
    mask_tokens = cache_tokens + (int(tail_ids.shape[-1]) if include_current_in_mask else 0)
    return model(
        input_ids=tail_ids,
        past_key_values=past_key_values,
        attention_mask=continuation_attention_mask(
            cache_tokens=mask_tokens,
            input_tokens=0,
            device=device,
        ),
        position_ids=continuation_mrope_position_ids(
            model=model,
            source_tokens=source_tokens,
            input_tokens=int(tail_ids.shape[-1]),
            device=device,
        ),
        use_cache=False,
        logits_to_keep=1,
    )


@torch.no_grad()
def score_vlm_full(
    *,
    model,
    processor,
    row: dict[str, Any],
    base_dir: str | Path,
    context_length: int,
    device: str,
    prompt_style: str = "compact",
    system_prompt: str = "",
) -> tuple[str, dict[str, float | list[float]]]:
    inputs = encode_vlm_prompt(
        processor=processor,
        row=row,
        base_dir=base_dir,
        device=device,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
    )
    source_inputs, tail_ids, source_tokens = split_source_and_tail(inputs)
    if source_tokens > context_length:
        return "", {"skipped_too_long": 1.0, "source_tokens": float(source_tokens)}
    source_outputs = model(**source_inputs, use_cache=True, logits_to_keep=1)
    label_ids = letter_token_ids(processor, choice_labels(len(row["choices"])), device=device)
    tail_outputs = _model_forward_tail(
        model=model,
        tail_ids=tail_ids,
        past_key_values=fresh_dynamic_cache(source_outputs.past_key_values),
        cache_tokens=source_tokens,
        source_tokens=source_tokens,
        device=device,
    )
    logits = choice_logits(tail_outputs.logits, label_ids)
    prediction = predict_letter_from_choice_logits(row, logits)
    return prediction, {
        "source_tokens": float(source_tokens),
        "full_letter_logits": [float(item) for item in logits.detach().cpu().reshape(-1)],
        "skipped_too_long": 0.0,
    }


@torch.no_grad()
def generate_vlm_full(
    *,
    model,
    processor,
    row: dict[str, Any],
    base_dir: str | Path,
    context_length: int,
    device: str,
    prompt_style: str = "official_mmmu",
    system_prompt: str = "",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
) -> tuple[str, dict[str, float | str]]:
    inputs = encode_vlm_prompt(
        processor=processor,
        row=row,
        base_dir=base_dir,
        device=device,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
    )
    source_tokens = int(inputs["input_ids"].shape[-1])
    if source_tokens > context_length:
        return "", {"skipped_too_long": 1.0, "source_tokens": float(source_tokens)}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )
    outputs = model.generate(**inputs, **generation_kwargs)
    generated_ids = outputs[0, source_tokens:]
    generated_text = processor.batch_decode(
        [generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    prediction = extract_vlm_answer_letter(generated_text, row)
    return prediction, {
        "source_tokens": float(source_tokens),
        "generation_tokens": float(generated_ids.numel()),
        "parse_error": float(prediction == ""),
        "generated_text": generated_text,
        "skipped_too_long": 0.0,
    }


def score_vlm_full_and_compact(
    *,
    model,
    processor,
    row: dict[str, Any],
    base_dir: str | Path,
    context_length: int,
    device: str,
    compactor: StillCompactor,
    prompt_style: str = "compact",
    system_prompt: str = "",
) -> tuple[str, str, dict[str, float | list[float]]]:
    inputs = encode_vlm_prompt(
        processor=processor,
        row=row,
        base_dir=base_dir,
        device=device,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
    )
    source_inputs, tail_ids, source_tokens = split_source_and_tail(inputs)
    if source_tokens > context_length:
        return "", "", {"skipped_too_long": 1.0, "source_tokens": float(source_tokens)}

    with torch.no_grad():
        source_outputs = model(**source_inputs, use_cache=True, logits_to_keep=1)
        label_ids = letter_token_ids(processor, choice_labels(len(row["choices"])), device=device)
        full_outputs = _model_forward_tail(
            model=model,
            tail_ids=tail_ids,
            past_key_values=fresh_dynamic_cache(source_outputs.past_key_values),
            cache_tokens=source_tokens,
            source_tokens=source_tokens,
            device=device,
        )
        full_logits = choice_logits(full_outputs.logits, label_ids)

    compact_cache = compactor(
        source_outputs.past_key_values,
        metadata={
            "source_tokens": source_tokens,
            "target_compression": source_tokens
            / max(getattr(compactor, "compact_tokens_per_layer", compactor.num_latents), 1),
        },
    )
    with still_biases(compact_cache.biases):
        compact_outputs = _model_forward_tail(
            model=model,
            tail_ids=tail_ids,
            past_key_values=compact_cache.as_dynamic_cache(),
            cache_tokens=compact_cache.num_tokens,
            source_tokens=source_tokens,
            device=device,
            include_current_in_mask=False,
        )
    compact_logits = choice_logits(compact_outputs.logits, label_ids)
    return (
        predict_letter_from_choice_logits(row, full_logits),
        predict_letter_from_choice_logits(row, compact_logits),
        {
            "source_tokens": float(source_tokens),
            "cache_tokens": float(compact_cache.num_tokens),
            "compression": float(compact_cache.compression_ratio_vs(source_tokens)),
            "full_letter_logits": [float(item) for item in full_logits.detach().cpu().reshape(-1)],
            "compact_letter_logits": [
                float(item) for item in compact_logits.detach().cpu().reshape(-1)
            ],
            "skipped_too_long": 0.0,
        },
    )


def vlm_training_forward(
    *,
    model,
    processor,
    compactor: StillCompactor,
    row: dict[str, Any],
    base_dir: str | Path,
    context_length: int,
    device: str,
    kl_weight: float,
    ce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    inputs = encode_vlm_prompt(processor=processor, row=row, base_dir=base_dir, device=device)
    source_inputs, tail_ids, source_tokens = split_source_and_tail(inputs)
    if source_tokens > context_length:
        zero = next(compactor.parameters()).new_tensor(0.0, requires_grad=True)
        return zero, {"skipped_too_long": 1.0, "source_tokens": float(source_tokens)}

    with torch.no_grad():
        source_outputs = model(**source_inputs, use_cache=True, logits_to_keep=1)
        label_ids = letter_token_ids(processor, choice_labels(len(row["choices"])), device=device)
        teacher_outputs = _model_forward_tail(
            model=model,
            tail_ids=tail_ids,
            past_key_values=fresh_dynamic_cache(source_outputs.past_key_values),
            cache_tokens=source_tokens,
            source_tokens=source_tokens,
            device=device,
        )
        teacher_logits = choice_logits(teacher_outputs.logits, label_ids).detach()

    compact_cache = compactor(
        source_outputs.past_key_values,
        metadata={"source_tokens": source_tokens},
    )
    with still_biases(compact_cache.biases):
        student_outputs = _model_forward_tail(
            model=model,
            tail_ids=tail_ids,
            past_key_values=compact_cache.as_dynamic_cache(),
            cache_tokens=compact_cache.num_tokens,
            source_tokens=source_tokens,
            device=device,
            include_current_in_mask=False,
        )
    student_logits = choice_logits(student_outputs.logits, label_ids)
    loss = letter_kl_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        target_index=int(row["answer_index"]),
        kl_weight=kl_weight,
        ce_weight=ce_weight,
    )
    with torch.no_grad():
        prediction = predict_letter_from_choice_logits(row, student_logits)
        gold = vlm_answer_letter(row)
    return loss, {
        "source_tokens": float(source_tokens),
        "cache_tokens": float(compact_cache.num_tokens),
        "compression": float(compact_cache.compression_ratio_vs(source_tokens)),
        "compact_accuracy": float(prediction == gold),
        "skipped_too_long": 0.0,
    }


@torch.no_grad()
def check_cache_equivalence(
    *,
    model,
    processor,
    row: dict[str, Any],
    base_dir: str | Path,
    context_length: int,
    device: str,
    prompt_style: str = "compact",
    system_prompt: str = "",
) -> float:
    inputs = encode_vlm_prompt(
        processor=processor,
        row=row,
        base_dir=base_dir,
        device=device,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
    )
    source_inputs, tail_ids, source_tokens = split_source_and_tail(inputs)
    if source_tokens > context_length:
        raise ValueError("Cannot check equivalence on a row longer than context_length")
    labels = choice_labels(len(row["choices"]))
    label_ids = letter_token_ids(processor, labels, device=device)
    direct_outputs = model(**inputs, use_cache=False, logits_to_keep=1)
    source_outputs = model(**source_inputs, use_cache=True, logits_to_keep=1)
    cached_outputs = _model_forward_tail(
        model=model,
        tail_ids=tail_ids,
        past_key_values=fresh_dynamic_cache(source_outputs.past_key_values),
        cache_tokens=source_tokens,
        source_tokens=source_tokens,
        device=device,
    )
    direct = choice_logits(direct_outputs.logits, label_ids)
    cached = choice_logits(cached_outputs.logits, label_ids)
    return float((direct - cached).abs().max().detach().cpu())


class VLMNeuralKVLightningModule(pl.LightningModule if pl is not None else nn.Module):
    """Lightning module for distilling a VLM prompt cache into a STILL compactor."""

    def __init__(
        self,
        *,
        model_name: str,
        compactor: dict[str, Any],
        context_length: int,
        train_base_dir: str,
        validation_base_dir: str,
        learning_rate: float = 5e-6,
        kl_weight: float = 1.0,
        ce_weight: float = 1.0,
        dtype: str = "bfloat16",
        model_load_device: str = "auto",
    ) -> None:
        if pl is None:
            raise ModuleNotFoundError("Install the train extra to use VLMNeuralKVLightningModule")
        super().__init__()
        self.save_hyperparameters()
        load_device = resolve_device(model_load_device)
        model, processor = load_vlm_model_and_processor(
            model_name,
            device=load_device,
            dtype=dtype_from_name(dtype),
        )
        object.__setattr__(self, "base_model", model)
        object.__setattr__(self, "processor", processor)
        self.compactor = StillCompactor.from_model_config(
            text_config_for_compactor(model.config),
            **compactor,
        )
        self._patched_attention_layers = 0

    def on_fit_start(self) -> None:
        self.base_model.to(self.device)
        self.base_model.eval()
        self._patched_attention_layers = enable_still_attention_bias(self.base_model)
        self.compactor.train()

    def on_validation_start(self) -> None:
        self.base_model.to(self.device)
        self.base_model.eval()
        if not self._patched_attention_layers:
            self._patched_attention_layers = enable_still_attention_bias(self.base_model)

    def training_step(self, batch: list[dict[str, Any]], batch_idx: int) -> torch.Tensor:
        del batch_idx
        device = _device_name(self.device)
        losses: list[torch.Tensor] = []
        metric_sums: dict[str, float] = {}
        for row in batch:
            loss, metrics = vlm_training_forward(
                model=self.base_model,
                processor=self.processor,
                compactor=self.compactor,
                row=row,
                base_dir=str(self.hparams.train_base_dir),
                context_length=int(self.hparams.context_length),
                device=device,
                kl_weight=float(self.hparams.kl_weight),
                ce_weight=float(self.hparams.ce_weight),
            )
            losses.append(loss)
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        mean_loss = torch.stack(losses).mean()
        batch_size = max(len(batch), 1)
        self.log("train/loss", mean_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size)
        for key, value in metric_sums.items():
            self.log(f"train/{key}", value / batch_size, on_step=True, on_epoch=False, batch_size=batch_size)
        return mean_loss

    def validation_step(self, batch: list[dict[str, Any]], batch_idx: int) -> None:
        del batch_idx
        if not batch:
            return
        row = batch[0]
        device = _device_name(self.device)
        full, compact, meta = score_vlm_full_and_compact(
            model=self.base_model,
            processor=self.processor,
            row=row,
            base_dir=str(self.hparams.validation_base_dir),
            context_length=int(self.hparams.context_length),
            device=device,
            compactor=self.compactor,
        )
        gold = vlm_answer_letter(row)
        self.log("val/full_accuracy", float(full == gold), on_epoch=True, batch_size=1)
        self.log("val/compact_accuracy", float(compact == gold), prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/compression", float(meta.get("compression", 0.0)), on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.compactor.parameters(), lr=float(self.hparams.learning_rate))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["neural_kv"] = {
            "model": self.hparams.model_name,
            "context_length": int(self.hparams.context_length),
            "compactor": dict(self.hparams.compactor),
            "modality": "vlm",
        }


def check_gate_summaries(gates: list[dict[str, Any]]) -> None:
    for gate in gates:
        path = Path(str(gate["summary_file"]))
        if not path.exists():
            raise SystemExit(f"Missing required full-cache gate summary: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows = int(metrics.get("rows", 0))
        full_accuracy = float(metrics.get("full_accuracy", 0.0))
        min_rows = int(gate.get("min_rows", 1))
        min_accuracy = float(gate.get("min_full_accuracy", 0.0))
        if rows < min_rows:
            raise SystemExit(f"Gate {path} has {rows} rows, expected at least {min_rows}")
        if full_accuracy < min_accuracy:
            raise SystemExit(
                f"Gate {path} full_accuracy={full_accuracy:.6f}, "
                f"expected >= {min_accuracy:.6f}"
            )
