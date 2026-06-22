#!/usr/bin/env python3
"""Evaluate a saved STILL checkpoint on MCQ rows."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

from neural_kv.data import answer_letter, read_jsonl
from neural_kv.models.checkpointing import parse_compactor_checkpoint
from neural_kv.models.compactor import StillCompactor
from neural_kv.modules.attention_bias import enable_still_attention_bias
from neural_kv.training.distillation import (
    dtype_from_name,
    generate_mcq_answer,
    generate_mcq_no_context_answer,
    load_model_and_tokenizer,
    resolve_device,
    score_mcq_full_and_compact_letters,
    score_mcq_letters,
    score_mcq_no_context,
)
from neural_kv.utils.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-model",
        default="",
        help="Optional HF model id or local path overriding checkpoint metadata.",
    )
    parser.add_argument("--eval-file", default="data/mcq/validation.jsonl")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many eval rows before scoring; useful for resumable shards.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Optional exclusive eval row index for sharded scoring.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--device-map-auto",
        action="store_true",
        help="Load the base model with Hugging Face device_map='auto' for multi-GPU eval.",
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
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Only score compact cache; skips no-context and full-context prompt scoring.",
    )
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--details-file",
        default="",
        help="Optional JSONL path to stream one scored row at a time.",
    )
    parser.add_argument(
        "--append-details",
        action="store_true",
        help="Append to --details-file instead of overwriting it.",
    )
    parser.add_argument(
        "--summary-file",
        default="",
        help="Optional path to write the pure JSON metrics summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_before = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage before eval: {storage_before}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = parse_compactor_checkpoint(checkpoint)
    model_name = spec.model_name
    context_length = spec.context_length

    device = resolve_device(args.device)
    eval_dtype = dtype_from_name(args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device,
        dtype=eval_dtype,
        device_map="auto" if args.device_map_auto else None,
    )
    patched_layers = enable_still_attention_bias(model)
    print(f"patched attention layers for STILL beta: {patched_layers}")

    compactor_dtype = eval_dtype if device.startswith("cuda") else torch.float32
    compactor = StillCompactor.from_model_config(
        model.config,
        **spec.compactor,
    ).to(device=device, dtype=compactor_dtype)
    compactor.load_state_dict(spec.state_dict)
    compactor.eval()

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index < 0:
        raise ValueError("--end-index must be non-negative")
    if args.end_index and args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    if args.end_index and args.limit and args.limit != args.end_index:
        raise ValueError("--limit and --end-index both set; use only --end-index for shards")
    row_limit = args.end_index or args.limit or None
    rows = read_jsonl(args.eval_file, limit=row_limit)[args.start_index :]
    details_handle = None
    if args.details_file:
        details_path = Path(args.details_file)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_handle = details_path.open("a" if args.append_details else "w", encoding="utf-8")
    compact_correct = 0
    full_correct = 0
    no_context_correct = 0
    compression_sum = 0.0
    compact_counts: collections.Counter[str] = collections.Counter()
    task_correct: collections.Counter[str] = collections.Counter()
    task_total: collections.Counter[str] = collections.Counter()
    details: list[dict[str, object]] = []
    use_chat_template = not args.no_chat_template

    for index, row in enumerate(rows):
        gold = answer_letter(row)
        task = str(row.get("ruler_task") or row.get("task") or row.get("source") or "unknown")
        if args.compact_only:
            if args.score_mode == "generation":
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
            no_context = ""
            full = ""
        elif args.score_mode == "generation":
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
            if args.score_mode in {"letter", "letter_delta"}:
                full, compact, meta = score_mcq_full_and_compact_letters(
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
            else:
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
        task_total[task] += 1
        task_correct[task] += int(compact == gold)
        compression_sum += meta["compression"]
        compact_counts[compact or "?"] += 1
        detail_row = {
            "index": args.start_index + index,
            "task": task,
            "gold": gold,
            "no_context": no_context,
            "full": full,
            "compact": compact,
            "source_tokens": float(meta.get("source_tokens", 0.0)),
            "cache_tokens": float(meta.get("cache_tokens", 0.0)),
            "compression": float(meta["compression"]),
            "used_chat_template": float(meta.get("used_chat_template", 0.0)),
        }
        for key in ("full_letter_logits", "compact_letter_logits"):
            if key in meta:
                detail_row[key] = meta[key]
        if args.details:
            details.append(detail_row)
        if details_handle is not None:
            details_handle.write(json.dumps(detail_row, sort_keys=True) + "\n")
            details_handle.flush()

    if details_handle is not None:
        details_handle.close()

    metrics = {
        "rows": len(rows),
        "compact_accuracy": compact_correct / len(rows) if rows else 0.0,
        "full_accuracy": full_correct / len(rows) if rows else 0.0,
        "no_context_accuracy": no_context_correct / len(rows) if rows else 0.0,
        "mean_compression": compression_sum / len(rows) if rows else 0.0,
        "compact_prediction_counts": dict(sorted(compact_counts.items())),
        "task_accuracy": {
            task: task_correct[task] / task_total[task]
            for task in sorted(task_total)
        },
        "task_counts": dict(sorted(task_total.items())),
    }
    if args.details:
        metrics["details"] = details
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    print(metrics_json)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(metrics_json + "\n", encoding="utf-8")
    storage_after = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage after eval: {storage_after}")


if __name__ == "__main__":
    main()
