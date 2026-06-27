#!/usr/bin/env python3
"""Evaluate Qwen3-VL full or compact-cache accuracy on VLM MCQ rows."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

from neural_kv.data.mcq import read_jsonl
from neural_kv.data.vlm import vlm_answer_letter
from neural_kv.models.checkpointing import parse_compactor_checkpoint
from neural_kv.models.compactor import StillCompactor
from neural_kv.modules.attention_bias import enable_still_attention_bias
from neural_kv.training.distillation import dtype_from_name, resolve_device
from neural_kv.training.vlm import (
    check_cache_equivalence,
    configure_vlm_processor_image_budget,
    generate_vlm_full,
    load_vlm_model_and_processor,
    score_vlm_full,
    score_vlm_full_and_compact,
    text_config_for_compactor,
)
from neural_kv.utils.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--mode", choices=["full", "compact"], default="full")
    parser.add_argument("--score-mode", choices=["letter_logprob", "generation"], default="letter_logprob")
    parser.add_argument(
        "--prompt-style",
        choices=["compact", "official_mmmu", "qwen_mmmu"],
        default="compact",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--max-storage", default="10TB")
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--details-file", default="")
    parser.add_argument("--min-full-accuracy", type=float, default=0.0)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--check-equivalence", action="store_true")
    parser.add_argument("--max-equivalence-diff", type=float, default=1e-3)
    parser.add_argument("--image-min-tokens", type=int, default=0)
    parser.add_argument("--image-max-tokens", type=int, default=0)
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


def _load_compactor(
    *,
    checkpoint_path: str,
    model,
    device: str,
    dtype: torch.dtype,
) -> tuple[StillCompactor, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = parse_compactor_checkpoint(checkpoint)
    compactor_dtype = dtype if device.startswith("cuda") else torch.float32
    compactor = StillCompactor.from_model_config(
        text_config_for_compactor(model.config),
        **spec.compactor,
    ).to(device=device, dtype=compactor_dtype)
    compactor.load_state_dict(spec.state_dict)
    compactor.eval()
    return compactor, int(spec.context_length)


def main() -> None:
    args = parse_args()
    storage_before = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage before vlm eval: {storage_before}")

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    model, processor = load_vlm_model_and_processor(
        args.model,
        device=device,
        dtype=dtype,
        device_map="auto" if args.device_map_auto else None,
    )
    image_budget = configure_vlm_processor_image_budget(
        processor,
        min_tokens=args.image_min_tokens,
        max_tokens=args.image_max_tokens,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    if image_budget:
        print(f"image processor size: {image_budget}")
    patched_layers = enable_still_attention_bias(model)
    print(f"patched attention layers for STILL beta: {patched_layers}")

    compactor = None
    context_length = int(args.context_length)
    if args.mode == "compact" and args.score_mode == "generation":
        raise SystemExit("--score-mode=generation is currently supported only with --mode=full")
    if args.mode == "compact":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --mode=compact")
        compactor, context_length = _load_compactor(
            checkpoint_path=args.checkpoint,
            model=model,
            device=device,
            dtype=dtype,
        )

    rows = read_jsonl(args.eval_file, limit=args.limit or None)
    base_dir = Path(args.eval_file).parent
    details_handle = None
    if args.details_file:
        details_path = Path(args.details_file)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_handle = details_path.open("w", encoding="utf-8")

    equivalence_diff = None
    if args.check_equivalence and rows:
        equivalence_diff = check_cache_equivalence(
            model=model,
            processor=processor,
            row=rows[0],
            base_dir=base_dir,
            context_length=context_length,
            device=device,
            prompt_style=args.prompt_style,
            system_prompt=args.system_prompt,
        )
        print(f"cache equivalence max label-logit diff: {equivalence_diff:.6g}")
        if equivalence_diff > args.max_equivalence_diff:
            raise SystemExit(
                f"Cache equivalence diff {equivalence_diff:.6g} exceeds "
                f"{args.max_equivalence_diff:.6g}"
            )

    full_correct = 0
    compact_correct = 0
    parsed_rows = 0
    parse_errors = 0
    skipped_too_long = 0
    compression_sum = 0.0
    prediction_counts: collections.Counter[str] = collections.Counter()
    task_correct: collections.Counter[str] = collections.Counter()
    task_total: collections.Counter[str] = collections.Counter()

    for index, row in enumerate(rows):
        try:
            gold = vlm_answer_letter(row)
            task = str(row.get("subject") or row.get("task") or row.get("source") or "unknown")
            if compactor is None:
                if args.score_mode == "generation":
                    full, meta = generate_vlm_full(
                        model=model,
                        processor=processor,
                        row=row,
                        base_dir=base_dir,
                        context_length=context_length,
                        device=device,
                        prompt_style=args.prompt_style,
                        system_prompt=args.system_prompt,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                    )
                else:
                    full, meta = score_vlm_full(
                        model=model,
                        processor=processor,
                        row=row,
                        base_dir=base_dir,
                        context_length=context_length,
                        device=device,
                        prompt_style=args.prompt_style,
                        system_prompt=args.system_prompt,
                    )
                compact = ""
            else:
                full, compact, meta = score_vlm_full_and_compact(
                    model=model,
                    processor=processor,
                    row=row,
                    base_dir=base_dir,
                    context_length=context_length,
                    device=device,
                    compactor=compactor,
                    prompt_style=args.prompt_style,
                    system_prompt=args.system_prompt,
                )
        except Exception as exc:  # noqa: BLE001 - stream details and continue scoring
            parse_errors += 1
            detail = {"index": index, "error": str(exc)}
            if details_handle is not None:
                details_handle.write(json.dumps(detail, sort_keys=True) + "\n")
            continue

        if meta.get("skipped_too_long"):
            skipped_too_long += 1
            continue
        if meta.get("parse_error"):
            parse_errors += 1
        parsed_rows += 1
        full_correct += int(full == gold)
        compact_correct += int(compact == gold)
        if compact:
            prediction_counts[compact] += 1
        else:
            prediction_counts[full or "?"] += 1
        task_total[task] += 1
        task_correct[task] += int((compact or full) == gold)
        compression_sum += float(meta.get("compression", 0.0))
        detail = {
            "index": index,
            "id": row.get("id", ""),
            "task": task,
            "gold": gold,
            "full": full,
            "compact": compact,
            **meta,
        }
        if details_handle is not None:
            details_handle.write(json.dumps(detail, sort_keys=True) + "\n")
            details_handle.flush()

    if details_handle is not None:
        details_handle.close()

    metrics = {
        "rows": parsed_rows,
        "full_accuracy": full_correct / parsed_rows if parsed_rows else 0.0,
        "compact_accuracy": compact_correct / parsed_rows if parsed_rows else 0.0,
        "mean_compression": compression_sum / parsed_rows if parsed_rows else 0.0,
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "parse_errors": parse_errors,
        "skipped_too_long": skipped_too_long,
        "task_accuracy": {
            task: task_correct[task] / task_total[task]
            for task in sorted(task_total)
        },
        "task_counts": dict(sorted(task_total.items())),
        "score_mode": args.score_mode,
        "prompt_style": args.prompt_style,
        "image_min_tokens": int(args.image_min_tokens),
        "image_max_tokens": int(args.image_max_tokens),
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "system_prompt": args.system_prompt,
    }
    if equivalence_diff is not None:
        metrics["cache_equivalence_max_diff"] = equivalence_diff

    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    print(metrics_json)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(metrics_json + "\n", encoding="utf-8")
    if parsed_rows < args.min_rows:
        raise SystemExit(f"Only scored {parsed_rows} rows; expected at least {args.min_rows}")
    if args.min_full_accuracy and metrics["full_accuracy"] < args.min_full_accuracy:
        raise SystemExit(
            f"full_accuracy={metrics['full_accuracy']:.6f} below "
            f"{args.min_full_accuracy:.6f}"
        )

    storage_after = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage after vlm eval: {storage_after}")


if __name__ == "__main__":
    main()
