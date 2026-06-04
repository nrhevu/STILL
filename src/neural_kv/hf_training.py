"""Hugging Face training/evaluation helpers for STILL."""

from __future__ import annotations

import re
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
    target_prefix: str
    used_chat_template: bool


SYSTEM_CONTEXT_TEMPLATE = (
    "Please answer the user's question using only the provided context.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Follow the requested answer format exactly.{thinking_instruction}"
)
RAW_TARGET_PREFIX = " "
CHAT_TARGET_PREFIX = ""
ANSWER_PATTERN = re.compile(
    r"(?:answer|final answer)\s*[:\-]?\s*[\(\[]?([ABCD])[\)\]]?",
    re.IGNORECASE,
)
TAIL_LETTER_PATTERN = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)")


def system_context_prompt(context: str, *, enable_thinking: bool = False) -> str:
    thinking_instruction = (
        "" if enable_thinking else " Do not emit <think> tags or chain-of-thought."
    )
    return SYSTEM_CONTEXT_TEMPLATE.format(
        context=context,
        thinking_instruction=thinking_instruction,
    )


def extract_answer_letter(text: str) -> str | None:
    answer_matches = list(ANSWER_PATTERN.finditer(text))
    if answer_matches:
        return answer_matches[-1].group(1).upper()

    tail = text[-256:]
    tail_matches = list(TAIL_LETTER_PATTERN.finditer(tail))
    if tail_matches:
        return tail_matches[-1].group(1).upper()
    return None


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


def _tokenize_text(
    tokenizer,
    text: str,
    *,
    device: str,
    add_special_tokens: bool = False,
) -> torch.Tensor:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    ).input_ids.to(device)


def _apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    template_args = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    for extra_args in (
        {"enable_thinking": enable_thinking},
        {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        {},
    ):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                **template_args,
                **extra_args,
            )
        except (TypeError, ValueError):
            continue
        if isinstance(rendered, str):
            return rendered
    return None


def _chat_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    device: str,
    enable_thinking: bool = False,
) -> torch.Tensor | None:
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if rendered is None:
        return None
    return _tokenize_text(
        tokenizer,
        rendered,
        device=device,
        add_special_tokens=False,
    )


def _build_chat_context(
    tokenizer,
    context: str,
    *,
    context_length: int,
    device: str,
    enable_thinking: bool = False,
) -> tuple[str, torch.Tensor] | None:
    def build_at_budget(token_budget: int) -> tuple[str, torch.Tensor] | None:
        context_text = tokenizer.decode(
            raw_context_ids[:token_budget].tolist(),
            skip_special_tokens=False,
        )
        system_prompt = system_context_prompt(context_text, enable_thinking=enable_thinking)
        system_ids = _chat_ids(
            tokenizer,
            [{"role": "system", "content": system_prompt}],
            add_generation_prompt=False,
            device=device,
            enable_thinking=enable_thinking,
        )
        if system_ids is None:
            return None
        return system_prompt, system_ids

    raw_context_ids = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    context_budget = min(int(raw_context_ids.shape[-1]), context_length)
    last_oversized_budget = context_budget + 1
    while context_budget >= 0:
        candidate = build_at_budget(context_budget)
        if candidate is None:
            return None
        system_prompt, system_ids = candidate
        if int(system_ids.shape[-1]) <= context_length:
            low_budget = context_budget
            high_budget = min(last_oversized_budget, int(raw_context_ids.shape[-1]) + 1)
            best = (system_prompt, system_ids)
            while low_budget + 1 < high_budget:
                midpoint = (low_budget + high_budget) // 2
                midpoint_candidate = build_at_budget(midpoint)
                if midpoint_candidate is None:
                    return None
                midpoint_prompt, midpoint_ids = midpoint_candidate
                if int(midpoint_ids.shape[-1]) <= context_length:
                    low_budget = midpoint
                    best = (midpoint_prompt, midpoint_ids)
                else:
                    high_budget = midpoint
            system_prompt, system_ids = best
            return system_prompt, system_ids
        overage = int(system_ids.shape[-1]) - context_length
        last_oversized_budget = context_budget
        context_budget -= max(overage + 16, 1)

    system_prompt = system_context_prompt("", enable_thinking=enable_thinking)
    system_ids = _chat_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        device=device,
        enable_thinking=enable_thinking,
    )
    if system_ids is None:
        return None
    return system_prompt, system_ids[:, :context_length]


def _encode_chat_continuation(
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    device: str,
    enable_thinking: bool = False,
) -> torch.Tensor | None:
    prefix_ids = _chat_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        device=device,
        enable_thinking=enable_thinking,
    )
    full_ids = _chat_ids(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        device=device,
        enable_thinking=enable_thinking,
    )
    if prefix_ids is None or full_ids is None:
        return None
    prefix_length = int(prefix_ids.shape[-1])
    if (
        int(full_ids.shape[-1]) >= prefix_length
        and torch.equal(full_ids[:, :prefix_length], prefix_ids)
    ):
        return full_ids[:, prefix_length:]

    prefix_text = _apply_chat_template(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    full_text = _apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prefix_text is None or full_text is None or not full_text.startswith(prefix_text):
        return None
    return _tokenize_text(
        tokenizer,
        full_text[len(prefix_text) :],
        device=device,
        add_special_tokens=False,
    )


def _encode_no_context_prompt(
    tokenizer,
    row: dict[str, object],
    *,
    device: str,
    use_chat_template: bool,
    enable_thinking: bool = False,
) -> tuple[torch.Tensor, str]:
    prompt = format_mcq_prompt(row, no_think=not enable_thinking)
    if use_chat_template:
        system_prompt = system_context_prompt("", enable_thinking=enable_thinking)
        prompt_ids = _chat_ids(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            device=device,
            enable_thinking=enable_thinking,
        )
        if prompt_ids is not None:
            return prompt_ids, CHAT_TARGET_PREFIX
    return _tokenize_text(
        tokenizer,
        prompt,
        device=device,
        add_special_tokens=False,
    ), RAW_TARGET_PREFIX


def _letter_ids(tokenizer, *, target_prefix: str, device: str) -> list[int]:
    ids: list[int] = []
    for label in "ABCD":
        label_ids = _tokenize_text(
            tokenizer,
            target_prefix + label,
            device=device,
            add_special_tokens=False,
        )
        if int(label_ids.shape[-1]) == 0:
            raise ValueError(f"Tokenizer produced no ids for answer label {label!r}")
        ids.append(int(label_ids[0, 0].item()))
    return ids


def _letter_index(row: dict[str, object]) -> int:
    return "ABCD".index(answer_letter(row))


def _pad_chat_context_ids(
    tokenizer,
    context_ids: torch.Tensor,
    *,
    context_length: int,
    device: str,
) -> torch.Tensor:
    missing = context_length - int(context_ids.shape[-1])
    if missing <= 0:
        return context_ids
    if missing > 8:
        return context_ids
    fill_ids = _tokenize_text(
        tokenizer,
        "\n",
        device=device,
        add_special_tokens=False,
    ).reshape(-1)
    if int(fill_ids.shape[-1]) == 0:
        fill_id = tokenizer.pad_token_id
        if fill_id is None:
            fill_id = tokenizer.eos_token_id
        if fill_id is None:
            return context_ids
        fill_ids = torch.tensor([fill_id], device=device, dtype=torch.long)
    repeats = (missing + int(fill_ids.shape[-1]) - 1) // int(fill_ids.shape[-1])
    padding = fill_ids.repeat(repeats)[:missing].unsqueeze(0)
    return torch.cat([context_ids, padding], dim=-1)


def encode_mcq(
    tokenizer,
    row: dict[str, object],
    *,
    context_length: int,
    device: str,
    target_mode: str = "choice_text",
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> EncodedMCQ:
    prompt = format_mcq_prompt(row, no_think=not enable_thinking)
    target_prefix = RAW_TARGET_PREFIX
    used_chat_template = False
    if use_chat_template:
        chat_context = _build_chat_context(
            tokenizer,
            str(row["context"]),
            context_length=context_length,
            device=device,
            enable_thinking=enable_thinking,
        )
        if chat_context is not None:
            system_prompt, context_ids = chat_context
            chat_prompt_ids = _encode_chat_continuation(
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=prompt,
                device=device,
                enable_thinking=enable_thinking,
            )
            if chat_prompt_ids is not None and int(chat_prompt_ids.shape[-1]) > 0:
                context_ids = _pad_chat_context_ids(
                    tokenizer,
                    context_ids,
                    context_length=context_length,
                    device=device,
                )
                prompt_ids = chat_prompt_ids
                target_prefix = CHAT_TARGET_PREFIX
                used_chat_template = True
            else:
                context_ids = _tokenize_text(
                    tokenizer,
                    str(row["context"]),
                    device=device,
                    add_special_tokens=True,
                )[:, :context_length]
                prompt_ids = _tokenize_text(
                    tokenizer,
                    prompt,
                    device=device,
                    add_special_tokens=False,
                )
        else:
            context_ids = _tokenize_text(
                tokenizer,
                str(row["context"]),
                device=device,
                add_special_tokens=True,
            )[:, :context_length]
            prompt_ids = _tokenize_text(
                tokenizer,
                prompt,
                device=device,
                add_special_tokens=False,
            )
    else:
        context_ids = _tokenize_text(
            tokenizer,
            str(row["context"]),
            device=device,
            add_special_tokens=True,
        )[:, :context_length]
        prompt_ids = _tokenize_text(
            tokenizer,
            prompt,
            device=device,
            add_special_tokens=False,
        )
    letter = answer_letter(row)
    if target_mode == "letter":
        target_text = letter
    elif target_mode == "choice_text":
        target_text = str(row["answer"])
    elif target_mode == "teacher_response":
        target_ids_value = row.get("teacher_response_token_ids")
        if isinstance(target_ids_value, list) and target_ids_value:
            target_ids = torch.tensor([target_ids_value], device=device, dtype=torch.long)
            return EncodedMCQ(
                context_ids=context_ids,
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                answer_letter=letter,
                target_prefix=target_prefix,
                used_chat_template=used_chat_template,
            )
        target_text = str(row.get("teacher_response") or "")
        if not target_text:
            raise ValueError("target_mode=teacher_response requires teacher_response text or ids")
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    prefix = target_prefix if target_mode != "teacher_response" else ""
    target_ids = _tokenize_text(
        tokenizer,
        prefix + target_text,
        device=device,
        add_special_tokens=False,
    )
    if target_ids.numel() == 0:
        target_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(letter)]], device=device)
    return EncodedMCQ(
        context_ids=context_ids,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        answer_letter=letter,
        target_prefix=target_prefix,
        used_chat_template=used_chat_template,
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


def _candidate_ids(
    tokenizer,
    choice: object,
    *,
    device: str,
    target_prefix: str,
) -> torch.Tensor:
    return _tokenize_text(
        tokenizer,
        target_prefix + str(choice),
        device=device,
        add_special_tokens=False,
    )


def _candidate_model_inputs(prompt_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    if candidate_ids.shape[-1] == 1:
        return prompt_ids
    return torch.cat([prompt_ids, candidate_ids[:, :-1]], dim=-1)


def _mean_candidate_logprob(
    *,
    logits: torch.Tensor,
    prompt_len: int,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    candidate_logits = answer_token_logits(logits, prompt_len, int(candidate_ids.shape[-1]))
    log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    token_scores = log_probs.gather(-1, candidate_ids.reshape(-1, 1)).squeeze(-1)
    return token_scores.mean()


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


def _letter_choice_logits(
    *,
    logits: torch.Tensor,
    prompt_len: int,
    label_ids: list[int],
) -> torch.Tensor:
    return logits[:, prompt_len - 1, label_ids].float()


def letter_kl_and_ce_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_index: int,
    kl_weight: float,
    ce_weight: float,
) -> torch.Tensor:
    """Distill the full-cache distribution over the four MCQ answer letters."""
    loss = student_logits.new_tensor(0.0, dtype=torch.float32)
    if kl_weight > 0:
        teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
        student_log_prob = torch.log_softmax(student_logits.float(), dim=-1)
        loss = loss + kl_weight * F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    if ce_weight > 0:
        target = torch.tensor([target_index], device=student_logits.device, dtype=torch.long)
        loss = loss + ce_weight * F.cross_entropy(
            student_logits.float(),
            target,
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
    target_mode: str,
    loss_mode: str = "token",
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_mcq(
        tokenizer,
        row,
        context_length=context_length,
        device=device,
        target_mode=target_mode,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    if loss_mode == "token":
        model_inputs = _continuation_inputs(encoded)
    elif loss_mode == "letter":
        model_inputs = encoded.prompt_ids
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
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
        if loss_mode == "token":
            teacher_logits = answer_token_logits(
                teacher_outputs.logits,
                prompt_len,
                target_len,
            ).detach()
        else:
            label_ids = _letter_ids(
                tokenizer,
                target_prefix=encoded.target_prefix,
                device=device,
            )
            teacher_logits = _letter_choice_logits(
                logits=teacher_outputs.logits,
                prompt_len=prompt_len,
                label_ids=label_ids,
            ).detach()

    compact_cache = compactor(
        full_outputs.past_key_values,
        metadata={
            "source_tokens": source_tokens,
            "target_compression": source_tokens
            / max(getattr(compactor, "compact_tokens_per_layer", compactor.num_latents), 1),
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
    if loss_mode == "token":
        student_logits = answer_token_logits(student_outputs.logits, prompt_len, target_len)
        loss = kl_and_ce_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_ids=encoded.target_ids,
            kl_weight=kl_weight,
            ce_weight=ce_weight,
        )
        gold_index = None
    else:
        student_logits = _letter_choice_logits(
            logits=student_outputs.logits,
            prompt_len=prompt_len,
            label_ids=label_ids,
        )
        gold_index = _letter_index(row)
        loss = letter_kl_and_ce_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_index=gold_index,
            kl_weight=kl_weight,
            ce_weight=ce_weight,
        )
    with torch.no_grad():
        if loss_mode == "token":
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
            teacher_choice = -1
            student_choice = -1
            gold_prob = float("nan")
        else:
            kl = letter_kl_and_ce_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                target_index=gold_index,
                kl_weight=1.0,
                ce_weight=0.0,
            )
            ce = letter_kl_and_ce_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                target_index=gold_index,
                kl_weight=0.0,
                ce_weight=1.0,
            )
            teacher_choice = int(torch.argmax(teacher_logits, dim=-1).item())
            student_choice = int(torch.argmax(student_logits, dim=-1).item())
            gold_prob = float(torch.softmax(student_logits.float(), dim=-1)[0, gold_index].cpu())
    metrics = {
        "kl": float(kl.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "loss_mode_letter": float(loss_mode == "letter"),
        "source_tokens": float(source_tokens),
        "compression": float(source_tokens / compact_cache.num_tokens),
        "used_chat_template": float(encoded.used_chat_template),
    }
    if loss_mode == "letter":
        metrics.update(
            {
                "teacher_choice": float(teacher_choice),
                "student_choice": float(student_choice),
                "student_gold_prob": gold_prob,
            }
        )
    return loss, metrics


@torch.no_grad()
def score_mcq_no_context(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    device: str,
    score_mode: str = "choice_loglik",
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> str:
    prompt_ids, target_prefix = _encode_no_context_prompt(
        tokenizer,
        row,
        device=device,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    if score_mode in {"letter", "letter_delta"}:
        label_ids = _letter_ids(tokenizer, target_prefix=target_prefix, device=device)
        outputs = model(input_ids=prompt_ids, use_cache=False)
        next_logits = outputs.logits[0, prompt_ids.shape[-1] - 1, label_ids].float()
        return "ABCD"[int(torch.argmax(next_logits).item())]
    if score_mode != "choice_loglik":
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    scores: list[float] = []
    for choice in row["choices"]:
        candidate_ids = _candidate_ids(
            tokenizer,
            choice,
            device=device,
            target_prefix=target_prefix,
        )
        model_inputs = _candidate_model_inputs(prompt_ids, candidate_ids)
        outputs = model(input_ids=model_inputs, use_cache=False)
        score = _mean_candidate_logprob(
            logits=outputs.logits,
            prompt_len=int(prompt_ids.shape[-1]),
            candidate_ids=candidate_ids,
        )
        scores.append(float(score.detach().cpu()))
    return "ABCD"[max(range(len(scores)), key=scores.__getitem__)]


@torch.no_grad()
def score_mcq_letters(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    context_length: int,
    device: str,
    compactor: StillCompactor | None = None,
    score_mode: str = "choice_loglik",
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> tuple[str, dict[str, float]]:
    encoded = encode_mcq(
        tokenizer,
        row,
        context_length=context_length,
        device=device,
        target_mode="letter",
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    source_tokens = int(encoded.context_ids.shape[-1])
    prompt_len = int(encoded.prompt_ids.shape[-1])

    if compactor is None:
        full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
        cache_tokens = source_tokens
        compact_cache = None
    else:
        full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
        compact_cache = compactor(
            full_outputs.past_key_values,
            metadata={"source_tokens": source_tokens},
        )
        cache_tokens = compact_cache.num_tokens

    if score_mode in {"letter", "letter_delta"}:
        label_ids = _letter_ids(
            tokenizer,
            target_prefix=encoded.target_prefix,
            device=device,
        )
        if compact_cache is None:
            outputs = model(
                input_ids=encoded.prompt_ids,
                past_key_values=full_outputs.past_key_values,
                attention_mask=_attention_mask(
                    source_tokens,
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
        else:
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
        next_logits = outputs.logits[0, prompt_len - 1, label_ids].float()
        if score_mode == "letter_delta":
            no_context_prompt_ids, no_context_prefix = _encode_no_context_prompt(
                tokenizer,
                row,
                device=device,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            no_context_label_ids = _letter_ids(
                tokenizer,
                target_prefix=no_context_prefix,
                device=device,
            )
            no_context_outputs = model(input_ids=no_context_prompt_ids, use_cache=False)
            no_context_logits = no_context_outputs.logits[
                0,
                no_context_prompt_ids.shape[-1] - 1,
                no_context_label_ids,
            ].float()
            next_logits = next_logits - no_context_logits
        winner = int(torch.argmax(next_logits).item())
    elif score_mode == "choice_loglik":
        scores: list[float] = []
        prefix_tokens = cache_tokens
        for choice in row["choices"]:
            candidate_ids = _candidate_ids(
                tokenizer,
                choice,
                device=device,
                target_prefix=encoded.target_prefix,
            )
            model_inputs = _candidate_model_inputs(encoded.prompt_ids, candidate_ids)
            if compact_cache is None:
                outputs = model(
                    input_ids=model_inputs,
                    past_key_values=full_outputs.past_key_values,
                    attention_mask=_attention_mask(
                        prefix_tokens,
                        model_inputs.shape[-1],
                        device=device,
                    ),
                    position_ids=_position_ids(
                        source_tokens,
                        model_inputs.shape[-1],
                        device=device,
                    ),
                    use_cache=False,
                )
            else:
                with still_biases(compact_cache.biases):
                    outputs = model(
                        input_ids=model_inputs,
                        past_key_values=compact_cache.as_dynamic_cache(),
                        attention_mask=_attention_mask(
                            prefix_tokens,
                            model_inputs.shape[-1],
                            device=device,
                        ),
                        position_ids=_position_ids(
                            source_tokens,
                            model_inputs.shape[-1],
                            device=device,
                        ),
                        use_cache=False,
                    )
            score = _mean_candidate_logprob(
                logits=outputs.logits,
                prompt_len=prompt_len,
                candidate_ids=candidate_ids,
            )
            scores.append(float(score.detach().cpu()))
        winner = max(range(len(scores)), key=scores.__getitem__)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    return "ABCD"[winner], {
        "source_tokens": float(source_tokens),
        "cache_tokens": float(cache_tokens),
        "compression": float(source_tokens / max(cache_tokens, 1)),
        "used_chat_template": float(encoded.used_chat_template),
    }


def _forward_with_optional_biases(model, biases: list[torch.Tensor] | None, **kwargs):
    if biases is None:
        return model(**kwargs)
    with still_biases(biases):
        return model(**kwargs)


@torch.no_grad()
def _greedy_generate(
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
    prompt_len = int(prompt_ids.shape[-1])
    if biases is not None:
        generated: list[int] = []
        eos_id = tokenizer.eos_token_id
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
            outputs = _forward_with_optional_biases(
                model,
                biases,
                input_ids=model_inputs,
                past_key_values=past_key_values,
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
        return tokenizer.decode(generated, skip_special_tokens=False)

    outputs = _forward_with_optional_biases(
        model,
        biases,
        input_ids=prompt_ids,
        past_key_values=past_key_values,
        attention_mask=_attention_mask(cache_tokens, prompt_len, device=prompt_ids.device),
        position_ids=_position_ids(source_position_start, prompt_len, device=prompt_ids.device),
        use_cache=True,
    )
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    eos_id = tokenizer.eos_token_id
    generated: list[int] = []

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if eos_id is not None and token_id == int(eos_id):
            break
        generated.append(token_id)
        past_tokens = cache_tokens + prompt_len + len(generated) - 1
        position = source_position_start + prompt_len + len(generated) - 1
        outputs = _forward_with_optional_biases(
            model,
            biases,
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=_attention_mask(past_tokens, 1, device=prompt_ids.device),
            position_ids=_position_ids(position, 1, device=prompt_ids.device),
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    return tokenizer.decode(generated, skip_special_tokens=False)


@torch.no_grad()
def generate_mcq_no_context_answer(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    device: str,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    max_new_tokens: int = 192,
) -> tuple[str | None, dict[str, object]]:
    prompt_ids, _ = _encode_no_context_prompt(
        tokenizer,
        row,
        device=device,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    text = _greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        past_key_values=None,
        cache_tokens=0,
        source_position_start=0,
        max_new_tokens=max_new_tokens,
    )
    return extract_answer_letter(text), {"generated_text": text}


@torch.no_grad()
def generate_mcq_answer(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    context_length: int,
    device: str,
    compactor: StillCompactor | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    max_new_tokens: int = 192,
) -> tuple[str | None, dict[str, object]]:
    encoded = encode_mcq(
        tokenizer,
        row,
        context_length=context_length,
        device=device,
        target_mode="letter",
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    source_tokens = int(encoded.context_ids.shape[-1])
    full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
    if compactor is None:
        past_key_values = full_outputs.past_key_values
        cache_tokens = source_tokens
        biases = None
    else:
        compact_cache = compactor(
            full_outputs.past_key_values,
            metadata={"source_tokens": source_tokens},
        )
        past_key_values = compact_cache.as_dynamic_cache()
        cache_tokens = compact_cache.num_tokens
        biases = compact_cache.biases

    text = _greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=encoded.prompt_ids,
        past_key_values=past_key_values,
        cache_tokens=cache_tokens,
        source_position_start=source_tokens,
        max_new_tokens=max_new_tokens,
        biases=biases,
    )
    return extract_answer_letter(text), {
        "generated_text": text,
        "source_tokens": float(source_tokens),
        "cache_tokens": float(cache_tokens),
        "compression": float(source_tokens / max(cache_tokens, 1)),
        "used_chat_template": float(encoded.used_chat_template),
    }
