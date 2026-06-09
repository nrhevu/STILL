#!/usr/bin/env python3
"""Synthetic needle-in-a-haystack evaluation for Hugging Face causal LMs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_kv.attention_bias import enable_still_attention_bias
from neural_kv.compactor import StillCompactor
from neural_kv.hf_training import (
    _build_chat_context,
    _encode_chat_continuation,
    _fresh_dynamic_cache,
    _greedy_generate,
    _tokenize_text,
    dtype_from_name,
    lexical_query_exact_token_indices,
    resolve_device,
)
from neural_kv.niah import (
    NiahCase,
    case_payload,
    make_niah_case,
    niah_question,
    niah_user_prompt,
    normalize_niah_answer,
)
from neural_kv.storage import check_storage_quota, default_storage_roots


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
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional train_still.py compactor checkpoint for compact-cache evaluation.",
    )
    parser.add_argument("--context-lengths", type=parse_csv_ints, default="4096,8192,16384,32768")
    parser.add_argument("--depths", type=parse_csv_floats, default="0,25,50,75,100")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--case-context-token-margin",
        type=int,
        default=0,
        help="Generate each needle context this many raw tokens shorter than the cache budget.",
    )
    parser.add_argument(
        "--split-cache",
        action="store_true",
        help="Evaluate the full model through context-cache + prompt splitting even without a compactor.",
    )
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


def load_compactor(
    *,
    checkpoint_path: str,
    model,
    model_name: str,
    device: str,
) -> tuple[StillCompactor, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = str(checkpoint.get("model"))
    if checkpoint_model != model_name:
        raise ValueError(
            f"checkpoint model {checkpoint_model!r} does not match requested model {model_name!r}"
        )
    compactor = StillCompactor.from_model_config(
        model.config,
        num_latents=int(checkpoint["num_latents"]),
        sink_tokens=int(checkpoint.get("sink_tokens", 0)),
        exact_tokens=int(checkpoint.get("exact_tokens", 0)),
        exact_strategy=str(checkpoint.get("exact_strategy", "prefix")),
        num_blocks=int(checkpoint.get("num_blocks", 2)),
        beta_base=str(checkpoint.get("beta_base", "log_compression")),
        layer_compactor_groups=int(checkpoint.get("layer_compactor_groups", 0)),
        head_specific_latents=bool(checkpoint.get("head_specific_latents", False)),
    ).to(device)
    compactor.load_state_dict(checkpoint["state_dict"])
    compactor.eval()
    return compactor, checkpoint


def encode_split_prompt(
    tokenizer,
    case: NiahCase,
    *,
    context_length: int,
    device: str,
    use_chat_template: bool,
    enable_thinking: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if use_chat_template:
        chat_context = _build_chat_context(
            tokenizer,
            case.context,
            context_length=context_length,
            device=device,
            enable_thinking=enable_thinking,
        )
        if chat_context is not None:
            system_prompt, context_ids = chat_context
            prompt_ids = _encode_chat_continuation(
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=niah_user_prompt(case, no_think=not enable_thinking),
                device=device,
                enable_thinking=enable_thinking,
            )
            if prompt_ids is not None and int(prompt_ids.shape[-1]) > 0:
                return context_ids, prompt_ids, True
    context_ids = _tokenize_text(
        tokenizer,
        case.context,
        device=device,
        add_special_tokens=True,
    )[:, :context_length]
    prompt_ids = _tokenize_text(
        tokenizer,
        "\n\n" + niah_user_prompt(case, no_think=not enable_thinking) + "\nAnswer:",
        device=device,
        add_special_tokens=False,
    )
    return context_ids, prompt_ids, False


@torch.no_grad()
def evaluate_case_full_prompt(
    model,
    tokenizer,
    case: NiahCase,
    *,
    device: str,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = (
        "You are a retrieval evaluator. Use only the provided context. "
        "Return only the requested secret value and no explanation.\n\n"
        f"<context>\n{case.context}\n</context>\n\n"
        f"/no_think\nQuestion: {niah_question(case)}\nAnswer with only the secret value."
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    started = time.time()
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
    return score_generated(case, generated, generated_tokens=int(generated_ids.shape[-1]), elapsed=elapsed) | {
        "prompt_tokens": int(input_ids.shape[-1]),
        "cache_tokens": int(input_ids.shape[-1]),
        "compression": 1.0,
        "mode": "full_prompt",
        "used_chat_template": False,
    }


@torch.no_grad()
def evaluate_case_split_cache(
    model,
    tokenizer,
    case: NiahCase,
    *,
    context_length: int,
    device: str,
    max_new_tokens: int,
    use_chat_template: bool,
    enable_thinking: bool,
    compactor: StillCompactor | None,
) -> dict[str, object]:
    context_ids, prompt_ids, used_chat_template = encode_split_prompt(
        tokenizer,
        case,
        context_length=context_length,
        device=device,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    source_tokens = int(context_ids.shape[-1])
    full_outputs = model(input_ids=context_ids, use_cache=True)
    cache_tokens = source_tokens
    biases = None
    past_key_values = full_outputs.past_key_values
    mode = "full_split_cache"
    if compactor is not None:
        exact_indices = None
        if int(getattr(compactor, "exact_tokens", 0)) > 0:
            exact_indices = lexical_query_exact_token_indices(
                tokenizer,
                {"context": case.context, "question": niah_question(case)},
                context_ids,
                max_tokens=int(getattr(compactor, "exact_tokens", 0)),
                device=device,
            )
        compact_cache = compactor(
            full_outputs.past_key_values,
            metadata={"source_tokens": source_tokens},
            exact_token_indices=exact_indices,
        )
        past_key_values = compact_cache.as_dynamic_cache()
        cache_tokens = compact_cache.num_tokens
        biases = compact_cache.biases
        mode = "compact_cache"
    started = time.time()
    generated = _greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        past_key_values=_fresh_dynamic_cache(past_key_values),
        cache_tokens=cache_tokens,
        source_position_start=source_tokens,
        max_new_tokens=max_new_tokens,
        biases=biases,
    ).strip()
    elapsed = time.time() - started
    generated_tokens = len(tokenizer(generated, add_special_tokens=False).input_ids)
    return score_generated(case, generated, generated_tokens=generated_tokens, elapsed=elapsed) | {
        "prompt_tokens": source_tokens + int(prompt_ids.shape[-1]),
        "source_tokens": source_tokens,
        "cache_tokens": cache_tokens,
        "compression": float(source_tokens / max(cache_tokens, 1)),
        "mode": mode,
        "used_chat_template": used_chat_template,
    }


def score_generated(
    case: NiahCase,
    generated: str,
    *,
    generated_tokens: int,
    elapsed: float,
) -> dict[str, object]:
    normalized_generated = normalize_niah_answer(generated)
    normalized_value = normalize_niah_answer(case.value)
    exact = normalized_generated == normalized_value
    contains = normalized_value in normalized_generated
    return {
        **case_payload(case),
        "generated_tokens": generated_tokens,
        "generated": generated,
        "exact_match": exact,
        "contains_match": contains,
        "success": exact or contains,
        "elapsed_seconds": elapsed,
    }


def summarize(records: list[dict[str, object]], *, model_name: str, checkpoint: str) -> dict[str, object]:
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
                "mean_cache_tokens": mean(float(item["cache_tokens"]) for item in items),
                "mean_compression": mean(float(item["compression"]) for item in items),
                "mean_elapsed_seconds": mean(float(item["elapsed_seconds"]) for item in items),
            }
        )
    return {
        "model": model_name,
        "checkpoint": checkpoint,
        "records": len(records),
        "overall_success_rate": mean(1.0 if item["success"] else 0.0 for item in records) if records else 0.0,
        "overall_exact_rate": mean(1.0 if item["exact_match"] else 0.0 for item in records) if records else 0.0,
        "overall_mean_compression": mean(float(item["compression"]) for item in records) if records else 0.0,
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
    compactor = None
    if args.checkpoint:
        patched_layers = enable_still_attention_bias(model)
        print(f"patched attention layers for STILL beta: {patched_layers}", flush=True)
        compactor, checkpoint = load_compactor(
            checkpoint_path=args.checkpoint,
            model=model,
            model_name=args.model,
            device=device,
        )
        checkpoint_context_length = int(checkpoint.get("context_length", 0))
        if len(args.context_lengths) == 1 and args.context_lengths[0] != checkpoint_context_length:
            print(
                f"warning: eval context length {args.context_lengths[0]} differs from "
                f"checkpoint context_length {checkpoint_context_length}",
                flush=True,
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
                    case_context_length = context_length - args.case_context_token_margin
                    if case_context_length <= 0:
                        raise ValueError("--case-context-token-margin must be smaller than each context length")
                    case = make_niah_case(
                        tokenizer,
                        context_length=case_context_length,
                        depth_percent=depth,
                        trial=trial,
                        seed=args.seed,
                    )
                    if compactor is None and not args.split_cache and not args.no_chat_template:
                        record = evaluate_case_full_prompt(
                            model,
                            tokenizer,
                            case,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        record = evaluate_case_split_cache(
                            model,
                            tokenizer,
                            case,
                            context_length=context_length,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                            use_chat_template=not args.no_chat_template,
                            enable_thinking=args.enable_thinking,
                            compactor=compactor,
                        )
                    if int(record["context_length"]) != context_length:
                        record["case_context_length"] = record["context_length"]
                        record["context_length"] = context_length
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    handle.flush()
                    print(
                        "context={context} depth={depth:g} trial={trial} success={success} "
                        "compression={compression:.3f} generated={generated!r}".format(
                            context=context_length,
                            depth=depth,
                            trial=trial,
                            success=record["success"],
                            compression=float(record["compression"]),
                            generated=str(record["generated"])[:120],
                        ),
                        flush=True,
                    )

    summary = summarize(records, model_name=args.model, checkpoint=args.checkpoint)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    after = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage after NIAH eval: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
