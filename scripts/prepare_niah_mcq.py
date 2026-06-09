#!/usr/bin/env python3
"""Build synthetic needle-in-a-haystack MCQ rows for compactor training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from neural_kv.niah import make_niah_case, niah_case_to_mcq_row
from neural_kv.storage import check_storage_quota, default_storage_roots


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
    parser.add_argument("--output-dir", default="data/niah_qwen3_32b_ctx8192")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument(
        "--raw-context-token-margin",
        type=int,
        default=1,
        help=(
            "Subtract this many tokens from generated raw context so raw-token training "
            "with add_special_tokens=True does not truncate the needle."
        ),
    )
    parser.add_argument("--depths", type=parse_csv_floats, default="0,25,50,75,100")
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--validation-rows", type=int, default=100)
    parser.add_argument("--test-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def build_rows(
    tokenizer,
    *,
    split: str,
    count: int,
    context_tokens: int,
    depths: list[float],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        depth = depths[index % len(depths)]
        case = make_niah_case(
            tokenizer,
            context_length=context_tokens,
            depth_percent=depth,
            trial=index,
            seed=seed,
        )
        row = niah_case_to_mcq_row(case, split=split, seed=seed)
        row["train_context_length"] = context_tokens
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    if args.context_length <= args.raw_context_token_margin:
        raise ValueError("--context-length must exceed --raw-context-token-margin")
    roots = default_storage_roots()
    print(f"storage before NIAH data prep: {check_storage_quota(roots, args.max_storage).summary()}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    context_tokens = args.context_length - args.raw_context_token_margin
    output_dir = Path(args.output_dir)
    splits = {
        "train": (args.train_rows, args.seed),
        "validation": (args.validation_rows, args.seed + 10_000),
        "test": (args.test_rows, args.seed + 20_000),
    }
    summary: dict[str, object] = {
        "model": args.model,
        "context_length": args.context_length,
        "generated_raw_context_tokens": context_tokens,
        "depths": args.depths,
        "splits": {},
    }
    for split, (count, seed) in splits.items():
        rows = build_rows(
            tokenizer,
            split=split,
            count=count,
            context_tokens=context_tokens,
            depths=args.depths,
            seed=seed,
        )
        written = write_jsonl(output_dir / f"{split}.jsonl", rows)
        summary["splits"][split] = written
        print(f"wrote {split}={written}")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"storage after NIAH data prep: {check_storage_quota(roots, args.max_storage).summary()}")


if __name__ == "__main__":
    main()
