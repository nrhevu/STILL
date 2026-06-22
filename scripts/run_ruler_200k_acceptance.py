#!/usr/bin/env python3
"""Run the RULER 200k coverage, evaluation, and strict target gates."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from neural_kv.utils.hf_cache import configure_hf_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--raw-file", default="data/ruler_200k/test.jsonl")
    parser.add_argument("--eval-file", default="data/ruler_200k/test.jsonl")
    parser.add_argument("--output-dir", default="outputs/ruler_200k_qwen3_235b_2507_8x")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--context-length", type=int, default=200000)
    parser.add_argument("--exact-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--score-mode", default="letter")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Score only the compressed cache path; skips full/no-context baselines.",
    )
    parser.add_argument(
        "--device-map-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Hugging Face device_map='auto' during checkpoint evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    output_dir = Path(args.output_dir)
    coverage_summary = output_dir / "coverage_summary.json"
    eval_summary = output_dir / "test_summary.json"
    eval_details = output_dir / "test_details.jsonl"
    max_new_tokens = getattr(args, "max_new_tokens", 192)
    no_chat_template = getattr(args, "no_chat_template", False)
    compact_only = getattr(args, "compact_only", False)
    return [
        [
            sys.executable,
            "scripts/check_ruler_coverage.py",
            "--input-file",
            str(args.raw_file),
            "--model",
            str(args.model),
            "--context-length",
            str(args.context_length),
            "--exact-tokens",
            str(args.exact_tokens),
            "--limit",
            str(args.limit),
            "--summary-file",
            str(coverage_summary),
        ],
        [
            sys.executable,
            "scripts/evaluate_checkpoint.py",
            "--checkpoint",
            str(args.checkpoint),
            *(["--base-model", str(args.base_model)] if args.base_model else []),
            "--eval-file",
            str(args.eval_file),
            "--limit",
            str(args.limit),
            "--score-mode",
            str(args.score_mode),
            "--max-new-tokens",
            str(max_new_tokens),
            *(["--no-chat-template"] if no_chat_template else []),
            "--device",
            str(args.device),
            "--dtype",
            str(args.dtype),
            *(["--device-map-auto"] if args.device_map_auto else []),
            *(["--compact-only"] if compact_only else []),
            "--details-file",
            str(eval_details),
            "--summary-file",
            str(eval_summary),
        ],
        [
            sys.executable,
            "scripts/check_ruler_200k_target.py",
            str(eval_summary),
            "--coverage-summary",
            str(coverage_summary),
            "--min-rows",
            str(args.limit),
            *(["--min-full-accuracy", "0"] if compact_only else []),
        ],
    ]


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    for command in commands:
        print("+ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
