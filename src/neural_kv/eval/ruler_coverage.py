#!/usr/bin/env python3
"""Check that RULER evidence is visible and fits lexical exact-token budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_kv.data.mcq import read_jsonl

SYSTEM_CONTEXT_TEMPLATE = (
    "Please answer the user's question using only the provided context.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Follow the requested answer format exactly.{thinking_instruction}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--context-length", type=int, default=200000)
    parser.add_argument("--exact-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--summary-file", default="")
    return parser.parse_args()


def _system_context_prompt(context: str) -> str:
    return SYSTEM_CONTEXT_TEMPLATE.format(
        context=context,
        thinking_instruction=" Do not emit <think> tags or chain-of-thought.",
    )


def _input_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(item) for item in input_ids]


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    limit = len(haystack) - len(needle) + 1
    for index in range(limit):
        if haystack[index] == first and haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _apply_chat_template(tokenizer: Any, context_text: str) -> str | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    messages = [{"role": "system", "content": _system_context_prompt(context_text)}]
    template_args = {"tokenize": False, "add_generation_prompt": False}
    for extra_args in (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    ):
        try:
            rendered = tokenizer.apply_chat_template(messages, **template_args, **extra_args)
        except (TypeError, ValueError):
            continue
        if isinstance(rendered, str):
            return rendered
    return None


def _chat_context_budget(
    tokenizer: Any,
    raw_ids: list[int],
    *,
    context_length: int,
    required_raw_tokens: int | None = None,
) -> int:
    if required_raw_tokens is not None:
        required = max(0, min(int(required_raw_tokens), len(raw_ids)))
        rendered = _apply_chat_template(tokenizer, tokenizer.decode(raw_ids[:required]))
        if rendered is None:
            return min(len(raw_ids), context_length)
        rendered_tokens = len(_input_ids(tokenizer, rendered))
        if rendered_tokens > context_length:
            return max(0, required - 1)
        template_overhead = max(0, rendered_tokens - required)
        return min(len(raw_ids), max(required, context_length - template_overhead))

    high = min(len(raw_ids), context_length)
    rendered = _apply_chat_template(tokenizer, tokenizer.decode(raw_ids[:high]))
    if rendered is None:
        return high
    if len(_input_ids(tokenizer, rendered)) <= context_length:
        return high

    low = 0
    while low + 1 < high:
        midpoint = (low + high) // 2
        rendered = _apply_chat_template(tokenizer, tokenizer.decode(raw_ids[:midpoint]))
        if rendered is None:
            return midpoint
        if len(_input_ids(tokenizer, rendered)) <= context_length:
            low = midpoint
        else:
            high = midpoint
    return low


def row_coverage(
    tokenizer: Any,
    row: dict[str, object],
    *,
    context_length: int,
    exact_tokens: int,
    use_chat_template: bool = True,
) -> dict[str, object]:
    context = str(row.get("context", ""))
    target_line = str(row.get("target_line") or row.get("answer") or "")
    raw_ids = _input_ids(tokenizer, context, add_special_tokens=False)

    target_ids: list[int] = []
    target_start = -1
    for candidate in (
        target_line,
        "\n" + target_line,
        target_line + "\n",
        "\n" + target_line + "\n",
    ):
        candidate_ids = _input_ids(tokenizer, candidate, add_special_tokens=False)
        found = _find_subsequence(raw_ids, candidate_ids)
        if found >= 0:
            target_ids = candidate_ids
            target_start = found
            break
    target_end = target_start + len(target_ids) if target_start >= 0 else -1
    context_budget = (
        _chat_context_budget(
            tokenizer,
            raw_ids,
            context_length=context_length,
            required_raw_tokens=target_end if target_end >= 0 else None,
        )
        if use_chat_template
        else min(len(_input_ids(tokenizer, context, add_special_tokens=True)), context_length)
    )
    failures: list[str] = []
    if target_start < 0:
        failures.append("target_line_not_found")
    if target_ids and len(target_ids) > exact_tokens:
        failures.append("target_line_exceeds_exact_tokens")
    if target_end > context_budget:
        failures.append("target_line_truncated")
    if not target_ids:
        failures.append("target_line_has_no_tokens")

    return {
        "id": row.get("id", ""),
        "task": row.get("ruler_task") or row.get("task") or row.get("source") or "unknown",
        "target_token_count": len(target_ids),
        "target_token_start": target_start,
        "target_token_end": target_end,
        "context_budget_tokens": context_budget,
        "raw_context_tokens": len(raw_ids),
        "passes": not failures,
        "failures": failures,
    }


def summarize_coverage(results: list[dict[str, object]]) -> dict[str, object]:
    failures = [result for result in results if not result.get("passes")]
    target_lengths = [int(result["target_token_count"]) for result in results]
    margins = [
        int(result["context_budget_tokens"]) - int(result["target_token_end"])
        for result in results
        if int(result.get("target_token_end", -1)) >= 0
    ]
    return {
        "rows": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "max_target_token_count": max(target_lengths) if target_lengths else 0,
        "min_context_margin_tokens": min(margins) if margins else None,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    from neural_kv.utils.hf_cache import configure_hf_cache

    configure_hf_cache()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rows = read_jsonl(args.input_file, limit=(args.limit or None))
    results = [
        row_coverage(
            tokenizer,
            row,
            context_length=args.context_length,
            exact_tokens=args.exact_tokens,
            use_chat_template=not args.no_chat_template,
        )
        for row in rows
    ]
    summary = summarize_coverage(results)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(payload + "\n", encoding="utf-8")
    if summary["failed"]:
        raise SystemExit(f"RULER coverage failed for {summary['failed']} rows")


if __name__ == "__main__":
    main()
