#!/usr/bin/env python3
"""Train a STILL compactor against full-cache teacher outputs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from neural_kv.attention_bias import enable_still_attention_bias
from neural_kv.compactor import StillCompactor
from neural_kv.data import answer_letter, read_jsonl
from neural_kv.hf_training import (
    dtype_from_name,
    load_model_and_tokenizer,
    resolve_device,
    score_mcq_letters,
    score_mcq_no_context,
    training_forward,
)
from neural_kv.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--train-file", default="data/mcq/train.jsonl")
    parser.add_argument("--eval-file", default="data/mcq/validation.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/smoke")
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--target-mode", choices=["choice_text", "letter"], default="letter")
    parser.add_argument(
        "--score-mode",
        choices=["choice_loglik", "letter"],
        default="letter",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize context and prompt as raw text instead of chat-template system/user turns.",
    )
    parser.add_argument("--latent-dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def save_checkpoint(
    *,
    path: Path,
    compactor: StillCompactor,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": args.model,
            "num_latents": args.num_latents,
            "num_blocks": args.num_blocks,
            "context_length": args.context_length,
            "latent_dropout": args.latent_dropout,
            "target_mode": args.target_mode,
            "score_mode": args.score_mode,
            "use_chat_template": not args.no_chat_template,
            "state_dict": compactor.state_dict(),
            "metrics": metrics,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    *,
    model,
    tokenizer,
    compactor: StillCompactor,
    rows: list[dict[str, object]],
    context_length: int,
    device: str,
    score_mode: str,
    use_chat_template: bool,
) -> dict[str, float]:
    if not rows:
        return {"compact_accuracy": 0.0, "full_accuracy": 0.0}
    compact_correct = 0
    full_correct = 0
    no_context_correct = 0
    compression_sum = 0.0
    compactor.eval()
    for row in rows:
        gold = answer_letter(row)
        no_context_pred = score_mcq_no_context(
            model=model,
            tokenizer=tokenizer,
            row=row,
            device=device,
            score_mode=score_mode,
            use_chat_template=use_chat_template,
        )
        full_pred, _ = score_mcq_letters(
            model=model,
            tokenizer=tokenizer,
            row=row,
            context_length=context_length,
            device=device,
            compactor=None,
            score_mode=score_mode,
            use_chat_template=use_chat_template,
        )
        compact_pred, compact_meta = score_mcq_letters(
            model=model,
            tokenizer=tokenizer,
            row=row,
            context_length=context_length,
            device=device,
            compactor=compactor,
            score_mode=score_mode,
            use_chat_template=use_chat_template,
        )
        no_context_correct += int(no_context_pred == gold)
        full_correct += int(full_pred == gold)
        compact_correct += int(compact_pred == gold)
        compression_sum += compact_meta["compression"]
    compactor.train()
    return {
        "compact_accuracy": compact_correct / len(rows),
        "full_accuracy": full_correct / len(rows),
        "no_context_accuracy": no_context_correct / len(rows),
        "mean_compression": compression_sum / len(rows),
        "used_chat_template": float(use_chat_template),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    roots = default_storage_roots()
    print(f"storage before training: {check_storage_quota(roots, args.max_storage).summary()}")

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    train_rows = read_jsonl(args.train_file, limit=(args.limit_train or None))
    eval_rows = read_jsonl(args.eval_file, limit=args.eval_limit)
    if not train_rows:
        raise ValueError(f"No training rows found in {args.train_file}")

    model, tokenizer = load_model_and_tokenizer(args.model, device=device, dtype=dtype)
    patched_layers = enable_still_attention_bias(model)
    print(f"patched attention layers for STILL beta: {patched_layers}")

    compactor = StillCompactor.from_model_config(
        model.config,
        num_latents=args.num_latents,
        num_blocks=args.num_blocks,
        latent_dropout=args.latent_dropout,
    ).to(device)
    compactor.train()
    optimizer = AdamW(compactor.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    schedule = [random.randrange(len(train_rows)) for _ in range(args.steps)]
    start_time = time.perf_counter()
    last_metrics: dict[str, float] = {}
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step, row_index in enumerate(tqdm(schedule, desc="training"), start=1):
            row = train_rows[row_index]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = training_forward(
                model=model,
                tokenizer=tokenizer,
                compactor=compactor,
                row=row,
                context_length=args.context_length,
                device=device,
                kl_weight=args.kl_weight,
                ce_weight=args.ce_weight,
                target_mode=args.target_mode,
                use_chat_template=not args.no_chat_template,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(compactor.parameters(), 1.0)
            optimizer.step()

            last_metrics = {"step": float(step), "loss": float(loss.detach().cpu()), **metrics}
            if args.eval_every and step % args.eval_every == 0:
                last_metrics.update(
                    evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        compactor=compactor,
                        rows=eval_rows,
                        context_length=args.context_length,
                        device=device,
                        score_mode=args.score_mode,
                        use_chat_template=not args.no_chat_template,
                    )
                )
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            metrics_file.flush()

            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    path=output_dir / f"step_{step}.pt",
                    compactor=compactor,
                    args=args,
                    step=step,
                    metrics=last_metrics,
                )
                check_storage_quota(roots, args.max_storage)

    last_metrics["train_seconds"] = time.perf_counter() - start_time
    if eval_rows:
        last_metrics.update(
            evaluate(
                model=model,
                tokenizer=tokenizer,
                compactor=compactor,
                rows=eval_rows,
                context_length=args.context_length,
                device=device,
                score_mode=args.score_mode,
                use_chat_template=not args.no_chat_template,
            )
        )
    save_checkpoint(
        path=output_dir / "final.pt",
        compactor=compactor,
        args=args,
        step=args.steps,
        metrics=last_metrics,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(last_metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(last_metrics, indent=2, sort_keys=True))
    print(f"storage after training: {check_storage_quota(roots, args.max_storage).summary()}")


if __name__ == "__main__":
    main()
