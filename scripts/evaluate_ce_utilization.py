#!/usr/bin/env python3
"""Evaluate compact-cache CE utilization on teacher-response trace rows."""

from __future__ import annotations

import argparse
import json

import torch

from neural_kv.attention_bias import enable_still_attention_bias
from neural_kv.compactor import StillCompactor
from neural_kv.data import read_jsonl
from neural_kv.hf_training import (
    dtype_from_name,
    load_model_and_tokenizer,
    resolve_device,
    trace_ce_scores,
)
from neural_kv.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-storage", default="10TB")
    parser.add_argument("--details", action="store_true")
    return parser.parse_args()


def _utilization(*, no_context: float, compact: float, full: float) -> float:
    denominator = no_context - full
    if abs(denominator) < 1e-12:
        return 0.0
    return (no_context - compact) / denominator


def main() -> None:
    args = parse_args()
    roots = default_storage_roots()
    print(f"storage before CE eval: {check_storage_quota(roots, args.max_storage).summary()}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model"])
    context_length = int(checkpoint["context_length"])
    num_latents = int(checkpoint["num_latents"])
    sink_tokens = int(checkpoint.get("sink_tokens", 0))
    exact_tokens = int(checkpoint.get("exact_tokens", 0))
    exact_strategy = str(checkpoint.get("exact_strategy", "prefix"))
    num_blocks = int(checkpoint.get("num_blocks", 2))
    layer_compactor_groups = int(checkpoint.get("layer_compactor_groups", 0))
    beta_base = str(checkpoint.get("beta_base", "log_compression"))

    rows = read_jsonl(args.eval_file, limit=(args.limit or None))
    if not rows:
        raise ValueError(f"No rows found in {args.eval_file}")
    missing = [
        index
        for index, row in enumerate(rows)
        if not row.get("teacher_response_token_ids") and not row.get("teacher_response")
    ]
    if missing:
        raise ValueError(
            "CE utilization requires teacher-response trace rows; "
            f"missing trace fields at row indices {missing[:5]}"
        )

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device,
        dtype=dtype_from_name(args.dtype),
    )
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
    ).to(device)
    compactor.load_state_dict(checkpoint["state_dict"])
    compactor.eval()

    token_total = 0.0
    sums = {
        "full_ce": 0.0,
        "compact_ce": 0.0,
        "compact_kl": 0.0,
        "no_context_ce": 0.0,
        "compression": 0.0,
    }
    details: list[dict[str, object]] = []
    use_chat_template = not args.no_chat_template

    for index, row in enumerate(rows):
        scores = trace_ce_scores(
            model=model,
            tokenizer=tokenizer,
            compactor=compactor,
            row=row,
            context_length=context_length,
            device=device,
            use_chat_template=use_chat_template,
            enable_thinking=args.enable_thinking,
        )
        weight = float(scores["target_tokens"])
        token_total += weight
        for key in ("full_ce", "compact_ce", "compact_kl", "no_context_ce"):
            sums[key] += scores[key] * weight
        sums["compression"] += scores["compression"]
        if args.details:
            detail = {"index": index}
            detail.update(scores)
            details.append(detail)

    metrics = {
        "rows": len(rows),
        "target_tokens": token_total,
        "full_ce": sums["full_ce"] / token_total,
        "compact_ce": sums["compact_ce"] / token_total,
        "compact_kl": sums["compact_kl"] / token_total,
        "no_context_ce": sums["no_context_ce"] / token_total,
        "mean_compression": sums["compression"] / len(rows),
    }
    metrics["ce_utilization"] = _utilization(
        no_context=metrics["no_context_ce"],
        compact=metrics["compact_ce"],
        full=metrics["full_ce"],
    )
    if args.details:
        metrics["details"] = details
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"storage after CE eval: {check_storage_quota(roots, args.max_storage).summary()}")


if __name__ == "__main__":
    main()
