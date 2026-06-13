#!/usr/bin/env python3
"""Evaluate a saved STILL checkpoint on MCQ rows."""

from __future__ import annotations

import argparse
import collections
import json

import torch

from neural_kv.attention_bias import enable_still_attention_bias
from neural_kv.compactor import StillCompactor
from neural_kv.data import answer_letter, read_jsonl
from neural_kv.hf_training import (
    dtype_from_name,
    generate_mcq_answer,
    generate_mcq_no_context_answer,
    infer_input_device,
    load_model_and_tokenizer,
    place_compactor_for_model,
    resolve_device,
    score_mcq_letters,
    score_mcq_no_context,
)
from neural_kv.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-file", default="data/mcq/validation.jsonl")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional Hugging Face device_map value, e.g. 'auto' for multi-GPU loading.",
    )
    parser.add_argument(
        "--max-memory",
        default="",
        help="Optional device memory map, e.g. '0=280GiB,1=280GiB,cpu=512GiB'.",
    )
    parser.add_argument(
        "--rope-scaling",
        default="",
        help="Optional JSON rope_scaling override passed through AutoConfig.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=0,
        help="Optional max_position_embeddings override for long-context YaRN runs.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["choice_loglik", "letter", "letter_delta", "generation"],
        default="letter",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Maximum generated tokens per MCQ when --score-mode=generation.",
    )
    parser.add_argument("--max-storage", default="10TB")
    parser.add_argument("--details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_before = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage before eval: {storage_before}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model"])
    context_length = int(checkpoint["context_length"])
    num_latents = int(checkpoint["num_latents"])
    sink_tokens = int(checkpoint.get("sink_tokens", 0))
    exact_tokens = int(checkpoint.get("exact_tokens", 0))
    exact_strategy = str(checkpoint.get("exact_strategy", "prefix"))
    num_blocks = int(checkpoint.get("num_blocks", 2))
    layer_compactor_groups = int(checkpoint.get("layer_compactor_groups", 0))
    head_specific_latents = bool(checkpoint.get("head_specific_latents", False))
    beta_base = str(checkpoint.get("beta_base", "log_compression"))

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device,
        dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory=args.max_memory,
        rope_scaling=args.rope_scaling,
        max_position_embeddings=args.max_position_embeddings,
    )
    device = infer_input_device(model, fallback=device)
    print(f"model input device: {device}")
    patched_layers = enable_still_attention_bias(model)
    print(f"patched attention layers for STILL beta: {patched_layers}")

    compactor = StillCompactor.from_model_config(
        model.config,
        num_latents=num_latents,
        sink_tokens=sink_tokens,
        exact_tokens=exact_tokens,
        exact_strategy=exact_strategy,
        num_blocks=num_blocks,
        beta_base=beta_base,
        layer_compactor_groups=layer_compactor_groups,
        head_specific_latents=head_specific_latents,
    )
    compactor.load_state_dict(checkpoint["state_dict"])
    placement = place_compactor_for_model(compactor, model, fallback_device=device)
    print(f"compactor placement devices: {sorted(set(placement.values()))}")
    compactor.eval()

    rows = read_jsonl(args.eval_file, limit=args.limit)
    compact_correct = 0
    full_correct = 0
    no_context_correct = 0
    compression_sum = 0.0
    compact_counts: collections.Counter[str] = collections.Counter()
    details: list[dict[str, object]] = []
    use_chat_template = not args.no_chat_template

    for index, row in enumerate(rows):
        gold = answer_letter(row)
        if args.score_mode == "generation":
            no_context, _ = generate_mcq_no_context_answer(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                use_chat_template=use_chat_template,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
            )
            full, _ = generate_mcq_answer(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=context_length,
                device=device,
                compactor=None,
                use_chat_template=use_chat_template,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
            )
            compact, meta = generate_mcq_answer(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=context_length,
                device=device,
                compactor=compactor,
                use_chat_template=use_chat_template,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            no_context = score_mcq_no_context(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                score_mode=args.score_mode,
                use_chat_template=use_chat_template,
                enable_thinking=args.enable_thinking,
            )
            full, _ = score_mcq_letters(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=context_length,
                device=device,
                compactor=None,
                score_mode=args.score_mode,
                use_chat_template=use_chat_template,
                enable_thinking=args.enable_thinking,
            )
            compact, meta = score_mcq_letters(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=context_length,
                device=device,
                compactor=compactor,
                score_mode=args.score_mode,
                use_chat_template=use_chat_template,
                enable_thinking=args.enable_thinking,
            )
        no_context_correct += int(no_context == gold)
        full_correct += int(full == gold)
        compact_correct += int(compact == gold)
        compression_sum += meta["compression"]
        compact_counts[compact or "?"] += 1
        if args.details:
            details.append(
                {
                    "index": index,
                    "gold": gold,
                    "no_context": no_context,
                    "full": full,
                    "compact": compact,
                }
            )

    compact_accuracy = compact_correct / len(rows) if rows else 0.0
    full_accuracy = full_correct / len(rows) if rows else 0.0
    metrics = {
        "compact_accuracy": compact_accuracy,
        "full_accuracy": full_accuracy,
        "relative_accuracy_to_full": compact_accuracy / full_accuracy if full_accuracy > 0 else 0.0,
        "no_context_accuracy": no_context_correct / len(rows) if rows else 0.0,
        "mean_compression": compression_sum / len(rows) if rows else 0.0,
        "compact_prediction_counts": dict(sorted(compact_counts.items())),
    }
    if args.details:
        metrics["details"] = details
    print(json.dumps(metrics, indent=2, sort_keys=True))
    storage_after = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage after eval: {storage_after}")


if __name__ == "__main__":
    main()
