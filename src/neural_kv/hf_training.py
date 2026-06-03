"""Hugging Face training/evaluation helpers for STILL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_kv.attention_bias import still_biases
from neural_kv.compactor import StillCompactor
from neural_kv.data import answer_letter, format_mcq_prompt


@dataclass(frozen=True)
class EncodedMCQ:
    context_ids: torch.Tensor
    prompt_ids: torch.Tensor
    target_ids: torch.Tensor
    answer_letter: str


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def encode_mcq(
    tokenizer,
    row: dict[str, object],
    *,
    context_length: int,
    device: str,
) -> EncodedMCQ:
    context_ids = tokenizer(
        str(row["context"]),
        return_tensors="pt",
        truncation=True,
        max_length=context_length,
        add_special_tokens=True,
    ).input_ids.to(device)
    prompt = format_mcq_prompt(row)
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    letter = answer_letter(row)
    target_ids = tokenizer(
        " " + letter,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    if target_ids.numel() == 0:
        target_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(letter)]], device=device)
    return EncodedMCQ(
        context_ids=context_ids,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        answer_letter=letter,
    )


def _continuation_inputs(encoded: EncodedMCQ) -> torch.Tensor:
    if encoded.target_ids.shape[-1] == 1:
        return encoded.prompt_ids
    return torch.cat([encoded.prompt_ids, encoded.target_ids[:, :-1]], dim=-1)


def _position_ids(start: int, length: int, *, device: str) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


def _attention_mask(prefix_length: int, input_length: int, *, device: str) -> torch.Tensor:
    return torch.ones(1, prefix_length + input_length, device=device, dtype=torch.long)


def answer_token_logits(logits: torch.Tensor, prompt_len: int, target_len: int) -> torch.Tensor:
    start = prompt_len - 1
    end = start + target_len
    return logits[:, start:end, :].squeeze(0)


def kl_and_ce_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    kl_weight: float,
    ce_weight: float,
) -> torch.Tensor:
    loss = student_logits.new_tensor(0.0, dtype=torch.float32)
    if kl_weight > 0:
        teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
        student_log_prob = torch.log_softmax(student_logits.float(), dim=-1)
        loss = loss + kl_weight * F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    if ce_weight > 0:
        loss = loss + ce_weight * F.cross_entropy(
            student_logits.float(),
            target_ids.reshape(-1),
            reduction="mean",
        )
    return loss


def training_forward(
    *,
    model,
    tokenizer,
    compactor: StillCompactor,
    row: dict[str, object],
    context_length: int,
    device: str,
    kl_weight: float,
    ce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_mcq(tokenizer, row, context_length=context_length, device=device)
    model_inputs = _continuation_inputs(encoded)
    source_tokens = int(encoded.context_ids.shape[-1])
    prompt_len = int(encoded.prompt_ids.shape[-1])
    target_len = int(encoded.target_ids.shape[-1])

    with torch.no_grad():
        full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
        teacher_outputs = model(
            input_ids=model_inputs,
            past_key_values=full_outputs.past_key_values,
            attention_mask=_attention_mask(source_tokens, model_inputs.shape[-1], device=device),
            position_ids=_position_ids(source_tokens, model_inputs.shape[-1], device=device),
            use_cache=False,
        )
        teacher_logits = answer_token_logits(
            teacher_outputs.logits,
            prompt_len,
            target_len,
        ).detach()

    compact_cache = compactor(
        full_outputs.past_key_values,
        metadata={
            "source_tokens": source_tokens,
            "target_compression": source_tokens / compactor.num_latents,
        },
    )
    with still_biases(compact_cache.biases):
        student_outputs = model(
            input_ids=model_inputs,
            past_key_values=compact_cache.as_dynamic_cache(),
            attention_mask=_attention_mask(
                compact_cache.num_tokens,
                model_inputs.shape[-1],
                device=device,
            ),
            position_ids=_position_ids(source_tokens, model_inputs.shape[-1], device=device),
            use_cache=False,
        )
    student_logits = answer_token_logits(student_outputs.logits, prompt_len, target_len)
    loss = kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        target_ids=encoded.target_ids,
        kl_weight=kl_weight,
        ce_weight=ce_weight,
    )
    with torch.no_grad():
        kl = kl_and_ce_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_ids=encoded.target_ids,
            kl_weight=1.0,
            ce_weight=0.0,
        )
        ce = kl_and_ce_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_ids=encoded.target_ids,
            kl_weight=0.0,
            ce_weight=1.0,
        )
    return loss, {
        "kl": float(kl.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "source_tokens": float(source_tokens),
        "compression": float(source_tokens / compact_cache.num_tokens),
    }


@torch.no_grad()
def score_mcq_no_context(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    device: str,
) -> str:
    prompt_ids = tokenizer(
        format_mcq_prompt(row),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    label_ids = [
        tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids[0, 0].item()
        for label in "ABCD"
    ]
    outputs = model(input_ids=prompt_ids, use_cache=False)
    next_logits = outputs.logits[0, prompt_ids.shape[-1] - 1, label_ids].float()
    return "ABCD"[int(torch.argmax(next_logits).item())]


@torch.no_grad()
def score_mcq_letters(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    context_length: int,
    device: str,
    compactor: StillCompactor | None = None,
) -> tuple[str, dict[str, float]]:
    encoded = encode_mcq(tokenizer, row, context_length=context_length, device=device)
    source_tokens = int(encoded.context_ids.shape[-1])
    prompt_len = int(encoded.prompt_ids.shape[-1])
    label_ids = [
        tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids[0, 0].item()
        for label in "ABCD"
    ]

    if compactor is None:
        full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
        outputs = model(
            input_ids=encoded.prompt_ids,
            past_key_values=full_outputs.past_key_values,
            attention_mask=_attention_mask(
                source_tokens,
                encoded.prompt_ids.shape[-1],
                device=device,
            ),
            position_ids=_position_ids(source_tokens, encoded.prompt_ids.shape[-1], device=device),
            use_cache=False,
        )
        cache_tokens = source_tokens
    else:
        full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
        compact_cache = compactor(
            full_outputs.past_key_values,
            metadata={"source_tokens": source_tokens},
        )
        with still_biases(compact_cache.biases):
            outputs = model(
                input_ids=encoded.prompt_ids,
                past_key_values=compact_cache.as_dynamic_cache(),
                attention_mask=_attention_mask(
                    compact_cache.num_tokens,
                    encoded.prompt_ids.shape[-1],
                    device=device,
                ),
                position_ids=_position_ids(
                    source_tokens,
                    encoded.prompt_ids.shape[-1],
                    device=device,
                ),
                use_cache=False,
            )
        cache_tokens = compact_cache.num_tokens

    next_logits = outputs.logits[0, prompt_len - 1, label_ids].float()
    winner = int(torch.argmax(next_logits).item())
    return "ABCD"[winner], {
        "source_tokens": float(source_tokens),
        "cache_tokens": float(cache_tokens),
        "compression": float(source_tokens / max(cache_tokens, 1)),
    }
