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
    target_prefix: str
    used_chat_template: bool


SYSTEM_CONTEXT_TEMPLATE = (
    "Please answer the user's question using only the provided context.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Follow the requested answer format exactly. Do not emit <think> tags or chain-of-thought."
)
RAW_TARGET_PREFIX = " "
CHAT_TARGET_PREFIX = ""


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
) -> str | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    template_args = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    # Qwen3 uses this flag to add the empty no-thinking block that should precede
    # the final answer in the assistant turn.
    for extra_args in (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
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
) -> torch.Tensor | None:
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
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
) -> tuple[str, torch.Tensor] | None:
    def build_at_budget(token_budget: int) -> tuple[str, torch.Tensor] | None:
        context_text = tokenizer.decode(
            raw_context_ids[:token_budget].tolist(),
            skip_special_tokens=False,
        )
        system_prompt = SYSTEM_CONTEXT_TEMPLATE.format(context=context_text)
        system_ids = _chat_ids(
            tokenizer,
            [{"role": "system", "content": system_prompt}],
            add_generation_prompt=False,
            device=device,
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

    system_prompt = SYSTEM_CONTEXT_TEMPLATE.format(context="")
    system_ids = _chat_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        device=device,
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
) -> torch.Tensor | None:
    prefix_ids = _chat_ids(
        tokenizer,
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        device=device,
    )
    full_ids = _chat_ids(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        device=device,
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
    )
    full_text = _apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
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
) -> tuple[torch.Tensor, str]:
    prompt = format_mcq_prompt(row)
    if use_chat_template:
        system_prompt = SYSTEM_CONTEXT_TEMPLATE.format(context="")
        prompt_ids = _chat_ids(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            device=device,
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
) -> EncodedMCQ:
    prompt = format_mcq_prompt(row)
    target_prefix = RAW_TARGET_PREFIX
    used_chat_template = False
    if use_chat_template:
        chat_context = _build_chat_context(
            tokenizer,
            str(row["context"]),
            context_length=context_length,
            device=device,
        )
        if chat_context is not None:
            system_prompt, context_ids = chat_context
            chat_prompt_ids = _encode_chat_continuation(
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=prompt,
                device=device,
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
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    target_ids = _tokenize_text(
        tokenizer,
        target_prefix + target_text,
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
    use_chat_template: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_mcq(
        tokenizer,
        row,
        context_length=context_length,
        device=device,
        target_mode=target_mode,
        use_chat_template=use_chat_template,
    )
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
        "used_chat_template": float(encoded.used_chat_template),
    }


@torch.no_grad()
def score_mcq_no_context(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    device: str,
    score_mode: str = "choice_loglik",
    use_chat_template: bool = True,
) -> str:
    prompt_ids, target_prefix = _encode_no_context_prompt(
        tokenizer,
        row,
        device=device,
        use_chat_template=use_chat_template,
    )
    if score_mode == "letter":
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
) -> tuple[str, dict[str, float]]:
    encoded = encode_mcq(
        tokenizer,
        row,
        context_length=context_length,
        device=device,
        target_mode="letter",
        use_chat_template=use_chat_template,
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

    if score_mode == "letter":
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
