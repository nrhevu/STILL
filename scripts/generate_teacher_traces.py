#!/usr/bin/env python3
"""Generate full-cache teacher responses for MCQ distillation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from neural_kv.data import answer_letter, read_jsonl
from neural_kv.hf_training import (
    dtype_from_name,
    encode_mcq,
    load_model_and_tokenizer,
    resolve_device,
)
from neural_kv.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many input rows before generating traces.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append traces to --output-file instead of overwriting it.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def _attention_mask(total_length: int, *, device: str) -> torch.Tensor:
    return torch.ones(1, total_length, device=device, dtype=torch.long)


def _position_id(position: int, *, device: str) -> torch.Tensor:
    return torch.tensor([[position]], device=device, dtype=torch.long)


@torch.no_grad()
def generate_response(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    context_length: int,
    max_new_tokens: int,
    device: str,
    enable_thinking: bool,
) -> tuple[list[int], str]:
    encoded = encode_mcq(
        tokenizer,
        row,
        context_length=context_length,
        device=device,
        target_mode="letter",
        use_chat_template=True,
        enable_thinking=enable_thinking,
    )
    source_tokens = int(encoded.context_ids.shape[-1])
    prompt_tokens = int(encoded.prompt_ids.shape[-1])

    full_outputs = model(input_ids=encoded.context_ids, use_cache=True)
    outputs = model(
        input_ids=encoded.prompt_ids,
        past_key_values=full_outputs.past_key_values,
        attention_mask=_attention_mask(source_tokens + prompt_tokens, device=device),
        position_ids=torch.arange(
            source_tokens,
            source_tokens + prompt_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0),
        use_cache=True,
    )
    eos_id = tokenizer.eos_token_id
    generated: list[int] = []
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if eos_id is not None and token_id == int(eos_id):
            break
        generated.append(token_id)
        total_seen = source_tokens + prompt_tokens + len(generated)
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=_attention_mask(total_seen, device=device),
            position_ids=_position_id(total_seen - 1, device=device),
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    return generated, tokenizer.decode(generated, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    roots = default_storage_roots()
    print(f"storage before traces: {check_storage_quota(roots, args.max_storage).summary()}")
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype_from_name(args.dtype),
    )
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    rows = read_jsonl(args.input_file, limit=(args.limit or None))[args.start_index :]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct_tail = 0
    output_mode = "a" if args.append else "w"
    with output_path.open(output_mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="teacher traces"):
            token_ids, text = generate_response(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=args.context_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
                enable_thinking=args.enable_thinking,
            )
            payload = dict(row)
            payload["teacher_response"] = text
            payload["teacher_response_token_ids"] = token_ids
            payload["teacher_response_token_count"] = len(token_ids)
            payload["teacher_response_gold_letter"] = answer_letter(row)
            if answer_letter(row) in text[-16:]:
                correct_tail += 1
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()

    print(
        json.dumps(
            {
                "rows": len(rows),
                "start_index": args.start_index,
                "tail_contains_gold": correct_tail,
                "output_file": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"storage after traces: {check_storage_quota(roots, args.max_storage).summary()}")


if __name__ == "__main__":
    main()
