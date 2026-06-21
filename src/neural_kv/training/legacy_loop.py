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

from neural_kv.data import answer_letter, read_jsonl
from neural_kv.models.compactor import EXACT_TOKEN_STRATEGIES, StillCompactor
from neural_kv.modules.attention_bias import enable_still_attention_bias
from neural_kv.training.distillation import (
    dtype_from_name,
    generate_mcq_answer,
    generate_mcq_no_context_answer,
    load_model_and_tokenizer,
    resolve_device,
    score_mcq_letters,
    score_mcq_no_context,
    training_forward,
)
from neural_kv.utils.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--train-file", default="data/mcq/train.jsonl")
    parser.add_argument("--eval-file", default="data/mcq/validation.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/smoke")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional train_still.py checkpoint to initialize the compactor from.",
    )
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=0,
        help="Exact prefix KV tokens to prepend to the learned latent cache.",
    )
    parser.add_argument(
        "--exact-tokens",
        type=int,
        default=0,
        help="Additional exact KV tokens to prepend before learned latent slots.",
    )
    parser.add_argument(
        "--exact-strategy",
        choices=sorted(EXACT_TOKEN_STRATEGIES),
        default="prefix",
        help="Selection strategy for --exact-tokens.",
    )
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument(
        "--layer-compactor-groups",
        type=int,
        default=0,
        help="Number of shared depth compactor groups; 0 keeps one compactor per layer.",
    )
    parser.add_argument(
        "--head-specific-latents",
        action="store_true",
        help=(
            "Use a separate latent query table per KV head group instead of sharing "
            "one table across heads."
        ),
    )
    parser.add_argument(
        "--beta-base",
        choices=["zero", "log_compression"],
        default="zero",
        help=(
            "Initial beta offset. Baseten identity init uses zero; "
            "older checkpoints used log_compression."
        ),
    )
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of rows to accumulate before each optimizer step.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--reverse-kl-weight",
        type=float,
        default=0.0,
        help=(
            "Optional D_KL(student || teacher) weight for bidirectional "
            "distillation, inspired by KV-Distill-style objectives."
        ),
    )
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument(
        "--aux-letter-loss-weight",
        type=float,
        default=0.0,
        help="Optional weight for an auxiliary direct-letter distillation loss.",
    )
    parser.add_argument(
        "--aux-letter-enable-thinking",
        action="store_true",
        help="Use thinking-enabled formatting for the auxiliary direct-letter loss.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["token", "letter"],
        default="token",
        help="Use token-level KL/CE or candidate-letter KL/CE for MCQ training.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["choice_text", "letter", "teacher_response"],
        default="letter",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Use thinking-enabled chat formatting for training targets.",
    )
    parser.add_argument(
        "--eval-enable-thinking",
        action="store_true",
        help="Use thinking-enabled chat formatting during evaluation.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["choice_loglik", "letter", "letter_delta", "generation"],
        default="letter",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=192,
        help="Maximum generated tokens per MCQ when --score-mode=generation.",
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
    parser.add_argument(
        "--balanced-answer-sampling",
        action="store_true",
        help="Sample answer letters uniformly before sampling rows within each letter.",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "beta", "heads", "beta_heads", "latents", "latents_beta"],
        default="all",
        help="Restrict trainable compactor parameters for fine-tuning experiments.",
    )
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
            "sink_tokens": args.sink_tokens,
            "exact_tokens": args.exact_tokens,
            "exact_strategy": args.exact_strategy,
            "num_blocks": args.num_blocks,
            "layer_compactor_groups": args.layer_compactor_groups,
            "head_specific_latents": args.head_specific_latents,
            "beta_base": args.beta_base,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "kl_weight": args.kl_weight,
            "reverse_kl_weight": args.reverse_kl_weight,
            "ce_weight": args.ce_weight,
            "aux_letter_loss_weight": args.aux_letter_loss_weight,
            "aux_letter_enable_thinking": args.aux_letter_enable_thinking,
            "latent_dropout": args.latent_dropout,
            "loss_mode": args.loss_mode,
            "target_mode": args.target_mode,
            "score_mode": args.score_mode,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "init_checkpoint": args.init_checkpoint,
            "use_chat_template": not args.no_chat_template,
            "enable_thinking": args.enable_thinking,
            "eval_enable_thinking": args.eval_enable_thinking,
            "balanced_answer_sampling": args.balanced_answer_sampling,
            "trainable_scope": args.trainable_scope,
            "state_dict": compactor.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def expand_shared_latents_for_head_specific(
    state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expand older shared latent tables to the per-KV-head latent layout."""
    expanded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        target = target_state_dict.get(key)
        if (
            key.endswith(".latents")
            and target is not None
            and value.dim() == 2
            and target.dim() == 3
        ):
            expanded[key] = value.unsqueeze(0).expand(target.shape[0], -1, -1).clone()
        else:
            expanded[key] = value
    return expanded


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
    enable_thinking: bool,
    max_new_tokens: int,
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
        if score_mode == "generation":
            no_context_pred, _ = generate_mcq_no_context_answer(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
            full_pred, _ = generate_mcq_answer(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=context_length,
                device=device,
                compactor=None,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
            compact_pred, compact_meta = generate_mcq_answer(
                model=model,
                tokenizer=tokenizer,
                row=row,
                context_length=context_length,
                device=device,
                compactor=compactor,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
        else:
            no_context_pred = score_mcq_no_context(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                score_mode=score_mode,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
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
                enable_thinking=enable_thinking,
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
                enable_thinking=enable_thinking,
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
        sink_tokens=args.sink_tokens,
        exact_tokens=args.exact_tokens,
        exact_strategy=args.exact_strategy,
        num_blocks=args.num_blocks,
        latent_dropout=args.latent_dropout,
        beta_base=args.beta_base,
        layer_compactor_groups=args.layer_compactor_groups,
        head_specific_latents=args.head_specific_latents,
    ).to(device)
    initial_step = 0
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if str(checkpoint.get("model")) != args.model:
            raise ValueError(
                f"--init-checkpoint model {checkpoint.get('model')!r} does not match {args.model!r}"
            )
        if int(checkpoint.get("num_latents", -1)) != args.num_latents:
            raise ValueError("--init-checkpoint num_latents does not match --num-latents")
        if int(checkpoint.get("sink_tokens", 0)) != args.sink_tokens:
            raise ValueError("--init-checkpoint sink_tokens does not match --sink-tokens")
        if int(checkpoint.get("exact_tokens", 0)) != args.exact_tokens:
            raise ValueError("--init-checkpoint exact_tokens does not match --exact-tokens")
        checkpoint_exact_strategy = str(checkpoint.get("exact_strategy", "prefix"))
        if checkpoint_exact_strategy != args.exact_strategy:
            raise ValueError(
                f"--init-checkpoint exact_strategy {checkpoint_exact_strategy!r} "
                f"does not match --exact-strategy {args.exact_strategy!r}"
            )
        if int(checkpoint.get("num_blocks", -1)) != args.num_blocks:
            raise ValueError("--init-checkpoint num_blocks does not match --num-blocks")
        checkpoint_head_specific = bool(checkpoint.get("head_specific_latents", False))
        state_dict = checkpoint["state_dict"]
        if checkpoint_head_specific != args.head_specific_latents:
            if args.head_specific_latents and not checkpoint_head_specific:
                state_dict = expand_shared_latents_for_head_specific(
                    state_dict,
                    compactor.state_dict(),
                )
                print("expanded shared checkpoint latents to head-specific latent tables")
            else:
                raise ValueError(
                    "--init-checkpoint head_specific_latents does not match "
                    "--head-specific-latents"
                )
        checkpoint_groups = int(checkpoint.get("layer_compactor_groups", 0))
        if checkpoint_groups != args.layer_compactor_groups:
            raise ValueError(
                f"--init-checkpoint layer_compactor_groups {checkpoint_groups} "
                f"does not match --layer-compactor-groups {args.layer_compactor_groups}"
            )
        checkpoint_beta_base = str(checkpoint.get("beta_base", "log_compression"))
        if checkpoint_beta_base != args.beta_base:
            raise ValueError(
                f"--init-checkpoint beta_base {checkpoint_beta_base!r} "
                f"does not match --beta-base {args.beta_base!r}"
            )
        if int(checkpoint.get("context_length", -1)) != args.context_length:
            raise ValueError("--init-checkpoint context_length does not match --context-length")
        compactor.load_state_dict(state_dict)
        initial_step = int(checkpoint.get("step", 0))

    trainable_keywords = {
        "all": None,
        "beta": ("beta_head",),
        "heads": ("key_head", "value_head"),
        "beta_heads": ("beta_head", "key_head", "value_head"),
        "latents": ("latents",),
        "latents_beta": ("latents", "beta_head"),
    }[args.trainable_scope]
    trainable_parameters: list[torch.nn.Parameter] = []
    for name, parameter in compactor.named_parameters():
        is_trainable = trainable_keywords is None or any(
            keyword in name for keyword in trainable_keywords
        )
        parameter.requires_grad_(is_trainable)
        if is_trainable:
            trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise ValueError(f"No trainable parameters selected for scope {args.trainable_scope!r}")

    compactor.train()
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.aux_letter_loss_weight < 0:
        raise ValueError("--aux-letter-loss-weight must be non-negative")
    if args.reverse_kl_weight < 0:
        raise ValueError("--reverse-kl-weight must be non-negative")
    if args.balanced_answer_sampling:
        answer_groups: dict[str, list[int]] = {}
        for index, row in enumerate(train_rows):
            answer_groups.setdefault(answer_letter(row), []).append(index)
        letters = sorted(answer_groups)
        if not letters:
            raise ValueError("Cannot use --balanced-answer-sampling with no answer groups")
        schedule = [
            [random.choice(answer_groups[random.choice(letters)]) for _ in range(args.batch_size)]
            for _ in range(args.steps)
        ]
    else:
        schedule = [
            [random.randrange(len(train_rows)) for _ in range(args.batch_size)]
            for _ in range(args.steps)
        ]
    start_time = time.perf_counter()
    last_metrics: dict[str, float] = {}
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for local_step, row_indices in enumerate(tqdm(schedule, desc="training"), start=1):
            step = initial_step + local_step
            optimizer.zero_grad(set_to_none=True)
            metric_sums: dict[str, float] = {}
            loss_sum = 0.0
            for row_index in row_indices:
                row = train_rows[row_index]
                primary_loss, metrics = training_forward(
                    model=model,
                    tokenizer=tokenizer,
                    compactor=compactor,
                    row=row,
                    context_length=args.context_length,
                    device=device,
                    kl_weight=args.kl_weight,
                    ce_weight=args.ce_weight,
                    reverse_kl_weight=args.reverse_kl_weight,
                    target_mode=args.target_mode,
                    loss_mode=args.loss_mode,
                    use_chat_template=not args.no_chat_template,
                    enable_thinking=args.enable_thinking,
                )
                loss = primary_loss
                metrics["primary_loss"] = float(primary_loss.detach().cpu())
                if args.aux_letter_loss_weight > 0:
                    aux_loss, aux_metrics = training_forward(
                        model=model,
                        tokenizer=tokenizer,
                        compactor=compactor,
                        row=row,
                        context_length=args.context_length,
                        device=device,
                        kl_weight=args.kl_weight,
                        ce_weight=args.ce_weight,
                        reverse_kl_weight=args.reverse_kl_weight,
                        target_mode="letter",
                        loss_mode="letter",
                        use_chat_template=not args.no_chat_template,
                        enable_thinking=args.aux_letter_enable_thinking,
                    )
                    loss = loss + args.aux_letter_loss_weight * aux_loss
                    metrics["aux_letter_loss"] = float(aux_loss.detach().cpu())
                    for key, value in aux_metrics.items():
                        metrics[f"aux_letter_{key}"] = value
                (loss / args.batch_size).backward()
                loss_sum += float(loss.detach().cpu())
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            torch.nn.utils.clip_grad_norm_(compactor.parameters(), 1.0)
            optimizer.step()

            last_metrics = {
                "step": float(step),
                "batch_size": float(args.batch_size),
                "loss": loss_sum / args.batch_size,
                **{key: value / args.batch_size for key, value in metric_sums.items()},
            }
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
                        enable_thinking=args.eval_enable_thinking,
                        max_new_tokens=args.eval_max_new_tokens,
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
                enable_thinking=args.eval_enable_thinking,
                max_new_tokens=args.eval_max_new_tokens,
            )
        )
    final_step = initial_step + args.steps
    save_checkpoint(
        path=output_dir / "final.pt",
        compactor=compactor,
        args=args,
        step=final_step,
        metrics=last_metrics,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(last_metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(last_metrics, indent=2, sort_keys=True))
    print(f"storage after training: {check_storage_quota(roots, args.max_storage).summary()}")


if __name__ == "__main__":
    main()
