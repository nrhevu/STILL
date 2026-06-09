#!/usr/bin/env python3
"""Synthetic needle-in-a-haystack evaluation for Hugging Face causal LMs."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_kv.hf_training import dtype_from_name, resolve_device
from neural_kv.storage import check_storage_quota, default_storage_roots


@dataclass(frozen=True)
class NiahCase:
    context_length: int
    depth_percent: float
    trial: int
    key: str
    value: str
    context: str
    actual_context_tokens: int
    needle_char_offset: int


FILLER_TOPICS = [
    "quarterly ledger reconciliation",
    "warehouse climate inspection",
    "legacy invoice migration",
    "regional routing memo",
    "archival policy update",
    "supplier quality review",
    "call-center transcript digest",
    "maintenance ticket summary",
]


def parse_csv_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def parse_csv_floats(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one number")
    for item in items:
        if item < 0 or item > 100:
            raise argparse.ArgumentTypeError("depth percentages must be in [0, 100]")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--context-lengths", type=parse_csv_ints, default="4096,8192,16384,32768")
    parser.add_argument("--depths", type=parse_csv_floats, default="0,25,50,75,100")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--output", default="reports/niah_qwen3_32b.jsonl")
    parser.add_argument("--summary-output", default="reports/niah_qwen3_32b_summary.json")
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, object] = {
        "dtype": dtype if device.startswith("cuda") else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def filler_text_for_tokens(tokenizer, *, token_count: int, rng: random.Random) -> str:
    if token_count <= 0:
        return ""
    lines: list[str] = []
    ids: list[int] = []
    line_index = 0
    while len(ids) < token_count:
        chunk: list[str] = []
        for _ in range(128):
            topic = rng.choice(FILLER_TOPICS)
            marker = rng.randrange(10_000_000, 99_999_999)
            control = rng.randrange(100_000, 999_999)
            chunk.append(
                f"Archive line {line_index:05d}: topic={topic}; marker={marker}; "
                f"control value CV{control}; this sentence is unrelated to any secret key.\n"
            )
            line_index += 1
        lines.extend(chunk)
        ids = token_ids(tokenizer, "".join(lines))
    return tokenizer.decode(ids[:token_count], skip_special_tokens=False)


def make_case(
    tokenizer,
    *,
    context_length: int,
    depth_percent: float,
    trial: int,
    seed: int,
) -> NiahCase:
    rng = random.Random(seed + context_length * 10_003 + int(depth_percent * 100) * 97 + trial)
    key = f"NIAH-{context_length}-{int(depth_percent * 100):04d}-{trial}-{rng.randrange(1000, 9999)}"
    value = f"VX{rng.randrange(10**9, 10**10 - 1)}Z"
    needle = (
        f"\nIMPORTANT NEEDLE RECORD: The secret value for retrieval key {key} is {value}. "
        "When asked for this key, answer with only the secret value.\n"
    )
    needle_tokens = token_ids(tokenizer, needle)
    if len(needle_tokens) >= context_length:
        raise ValueError(
            f"Needle has {len(needle_tokens)} tokens, which does not fit context_length={context_length}"
        )
    filler_budget = context_length - len(needle_tokens)
    prefix_tokens = int(round(filler_budget * (depth_percent / 100.0)))
    suffix_tokens = filler_budget - prefix_tokens
    prefix = filler_text_for_tokens(tokenizer, token_count=prefix_tokens, rng=rng)
    suffix = filler_text_for_tokens(tokenizer, token_count=suffix_tokens, rng=rng)
    context = prefix + needle + suffix
    actual_context_tokens = len(token_ids(tokenizer, context))
    return NiahCase(
        context_length=context_length,
        depth_percent=depth_percent,
        trial=trial,
        key=key,
        value=value,
        context=context,
        actual_context_tokens=actual_context_tokens,
        needle_char_offset=context.find(needle.strip()),
    )


def apply_chat_template(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool) -> str | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    base = {"tokenize": False, "add_generation_prompt": True}
    for extra in (
        {"enable_thinking": enable_thinking},
        {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        {},
    ):
        try:
            rendered = tokenizer.apply_chat_template(messages, **base, **extra)
        except (TypeError, ValueError):
            continue
        if isinstance(rendered, str):
            return rendered
    return None


def build_prompt(tokenizer, case: NiahCase, *, use_chat_template: bool, enable_thinking: bool) -> str:
    system = (
        "You are a retrieval evaluator. Use only the provided context. "
        "Return only the requested secret value and no explanation."
    )
    if not enable_thinking:
        system += " Do not emit chain-of-thought or <think> tags."
    user = (
        f"<context>\n{case.context}\n</context>\n\n"
        f"Question: What is the secret value for retrieval key {case.key}?\n"
        "Answer with only the secret value."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if use_chat_template:
        rendered = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
        if rendered is not None:
            return rendered
    return f"{system}\n\n{user}\nAnswer:"


def normalize_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def evaluate_case(
    model,
    tokenizer,
    case: NiahCase,
    *,
    device: str,
    max_new_tokens: int,
    use_chat_template: bool,
    enable_thinking: bool,
) -> dict[str, object]:
    prompt = build_prompt(
        tokenizer,
        case,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    started = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    normalized_generated = normalize_text(generated)
    normalized_value = normalize_text(case.value)
    exact = normalized_generated == normalized_value
    contains = normalized_value in normalized_generated
    return {
        **asdict(case),
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": int(generated_ids.shape[-1]),
        "generated": generated,
        "exact_match": exact,
        "contains_match": contains,
        "success": exact or contains,
        "elapsed_seconds": elapsed,
    }


def summarize(records: list[dict[str, object]], *, model_name: str) -> dict[str, object]:
    groups: dict[tuple[int, float], list[dict[str, object]]] = {}
    for record in records:
        key = (int(record["context_length"]), float(record["depth_percent"]))
        groups.setdefault(key, []).append(record)
    by_context_depth = []
    for (context_length, depth), items in sorted(groups.items()):
        by_context_depth.append(
            {
                "context_length": context_length,
                "depth_percent": depth,
                "trials": len(items),
                "success_rate": mean(1.0 if item["success"] else 0.0 for item in items),
                "exact_rate": mean(1.0 if item["exact_match"] else 0.0 for item in items),
                "mean_prompt_tokens": mean(float(item["prompt_tokens"]) for item in items),
                "mean_elapsed_seconds": mean(float(item["elapsed_seconds"]) for item in items),
            }
        )
    return {
        "model": model_name,
        "records": len(records),
        "overall_success_rate": mean(1.0 if item["success"] else 0.0 for item in records) if records else 0.0,
        "overall_exact_rate": mean(1.0 if item["exact_match"] else 0.0 for item in records) if records else 0.0,
        "by_context_depth": by_context_depth,
    }


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    before = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage before NIAH eval: {before.summary()}", flush=True)

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
    )

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for context_length in args.context_lengths:
            for depth in args.depths:
                for trial in range(args.trials):
                    case = make_case(
                        tokenizer,
                        context_length=context_length,
                        depth_percent=depth,
                        trial=trial,
                        seed=args.seed,
                    )
                    record = evaluate_case(
                        model,
                        tokenizer,
                        case,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        use_chat_template=not args.no_chat_template,
                        enable_thinking=args.enable_thinking,
                    )
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    handle.flush()
                    print(
                        "context={context} depth={depth:g} trial={trial} success={success} "
                        "prompt_tokens={tokens} generated={generated!r}".format(
                            context=context_length,
                            depth=depth,
                            trial=trial,
                            success=record["success"],
                            tokens=record["prompt_tokens"],
                            generated=str(record["generated"])[:120],
                        ),
                        flush=True,
                    )

    summary = summarize(records, model_name=args.model)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    after = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage after NIAH eval: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
