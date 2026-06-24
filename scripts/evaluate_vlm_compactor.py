#!/usr/bin/env python3
"""Evaluate a neural KV compactor on vision-language benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from neural_kv.utils.config import load_config
from neural_kv.utils.rocm import apply_visible_device_for_idle_gpu, gpu_utilization


def _csv_ints(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(piece) for piece in value.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment/qwen3_vl_compactor_bench.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--details-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--resolutions", default="")
    parser.add_argument("--image-token-budgets", default="")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--target-compact-vs-full", type=float, default=0.0)
    parser.add_argument("--fail-under-target", action="store_true")
    parser.add_argument("--no-fail-under-target", action="store_true")
    parser.add_argument("--init-random-compactor", action="store_true")
    parser.add_argument("--no-wait-for-gpu", action="store_true")
    return parser.parse_args()


def _select_idle_gpu_or_wait(runtime: dict[str, Any]) -> None:
    preferred = int(runtime.get("preferred_gpu", 7))
    poll_seconds = int(runtime.get("gpu_poll_seconds", 60))
    wait = bool(runtime.get("wait_for_idle_gpu", True))
    while True:
        usage = gpu_utilization()
        print(f"ROCm utilization before VLM eval: {usage}")
        selected = apply_visible_device_for_idle_gpu(preferred=preferred, require_zero=True)
        if selected is not None:
            visible = os.environ["HIP_VISIBLE_DEVICES"]
            print(f"Selected idle GPU {selected}; HIP_VISIBLE_DEVICES={visible}")
            return
        if not wait:
            raise SystemExit("No 0% utilization GPU is available; refusing to start VLM eval")
        print(f"No 0% utilization GPU is available; waiting {poll_seconds}s")
        time.sleep(poll_seconds)


def _needs_gpu_guard(*, device: str, device_map_auto: bool, runtime: dict[str, Any]) -> bool:
    if not runtime.get("require_idle_gpu", False):
        return False
    if device == "cpu":
        return False
    return device in {"auto", "cuda"} or device.startswith("cuda") or device_map_auto


def _load_compactor(args, cfg, model, torch, model_name: str, device: str, dtype):
    from neural_kv.models.checkpointing import parse_compactor_checkpoint
    from neural_kv.models.compactor import StillCompactor

    benchmark = cfg.get("benchmark", {})
    checkpoint_path = args.checkpoint or benchmark.get("checkpoint", "")
    init_random = bool(args.init_random_compactor or benchmark.get("init_random_compactor", False))
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        spec = parse_compactor_checkpoint(checkpoint)
        compactor_config = spec.compactor
        state_dict = spec.state_dict
    elif init_random:
        compactor_config = cfg["model"]["compactor"]
        state_dict = None
    else:
        raise SystemExit(
            "Set --checkpoint to a trained compactor or --init-random-compactor for plumbing "
            "smoke tests. Random compactor results are not quality benchmarks."
        )

    compactor_dtype = dtype if device.startswith("cuda") else torch.float32
    compactor = StillCompactor.from_model_config(model.config, **compactor_config)
    if state_dict is not None:
        compactor.load_state_dict(state_dict)
    compactor.to(dtype=compactor_dtype)
    compactor.eval()
    for parameter in compactor.parameters():
        parameter.requires_grad_(False)
    print(
        "Loaded compactor for "
        f"{model_name}: latents={compactor_config.get('num_latents')} "
        f"sink={compactor_config.get('sink_tokens', 0)} "
        f"exact={compactor_config.get('exact_tokens', 0)}"
    )
    return compactor


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    benchmark = cfg.get("benchmark", {})
    model_cfg = cfg.get("model", {})

    device = args.device or str(runtime.get("model_load_device", "auto"))
    dtype_name = args.dtype or str(runtime.get("dtype", "bfloat16"))
    device_map_auto = bool(args.device_map_auto or runtime.get("device_map_auto", False))
    if args.no_wait_for_gpu:
        runtime = dict(runtime)
        runtime["wait_for_idle_gpu"] = False
    if _needs_gpu_guard(device=device, device_map_auto=device_map_auto, runtime=runtime):
        _select_idle_gpu_or_wait(runtime)

    import torch

    from neural_kv.data.vlm import iter_hf_vlm_examples_from_specs
    from neural_kv.eval.vlm_compactor import (
        evaluate_vlm_example,
        load_vlm_model,
        load_vlm_processor,
        model_input_device,
        summarize_vlm_results,
    )
    from neural_kv.modules.attention_bias import enable_still_attention_bias
    from neural_kv.training.distillation import dtype_from_name, resolve_device

    resolved_device = resolve_device(device)
    model_name = args.base_model or str(model_cfg.get("name", "Qwen/Qwen3-VL-8B-Instruct"))
    dtype = dtype_from_name(dtype_name)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    model = load_vlm_model(
        model_name,
        device=resolved_device,
        dtype=dtype,
        device_map="auto" if device_map_auto else None,
        trust_remote_code=trust_remote_code,
    )
    patched = enable_still_attention_bias(model)
    print(f"patched attention layers for STILL beta: {patched}")
    compactor = _load_compactor(args, cfg, model, torch, model_name, resolved_device, dtype)

    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/qwen3_vl_compactor_bench"))
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = Path(
        args.details_file or benchmark.get("details_file", output_dir / "details.jsonl")
    )
    summary_path = Path(
        args.summary_file or benchmark.get("summary_file", output_dir / "summary.json")
    )
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    limit = args.limit or int(benchmark.get("limit_per_dataset", 16))
    resolutions = _csv_ints(args.resolutions or benchmark.get("resolutions", [448]))
    budgets = _csv_ints(args.image_token_budgets or benchmark.get("image_token_budgets", [256]))
    max_new_tokens = args.max_new_tokens or int(benchmark.get("max_new_tokens", 32))
    no_think = bool(benchmark.get("no_think", True))
    target_compact_vs_full = float(
        args.target_compact_vs_full
        or benchmark.get("target_compact_vs_full_accuracy", 0.95)
    )
    fail_under_target = bool(benchmark.get("fail_under_target", False))
    if args.fail_under_target:
        fail_under_target = True
    if args.no_fail_under_target:
        fail_under_target = False
    dataset_specs = benchmark.get("datasets", [])
    if not dataset_specs:
        raise SystemExit("benchmark.datasets must contain at least one dataset spec")

    examples = list(iter_hf_vlm_examples_from_specs(dataset_specs, limit_per_dataset=limit))
    print(f"Loaded {len(examples)} VLM examples")
    input_device = model_input_device(model, fallback=resolved_device)
    results: list[dict[str, Any]] = []
    processor_cache: dict[int, Any] = {}
    with details_path.open("w", encoding="utf-8") as details_handle:
        for budget in budgets:
            processor_cache[budget] = load_vlm_processor(
                model_name,
                image_token_budget=budget,
                trust_remote_code=trust_remote_code,
            )
            for resolution in resolutions:
                for index, example in enumerate(examples):
                    row = evaluate_vlm_example(
                        model=model,
                        processor=processor_cache[budget],
                        example=example,
                        compactor=compactor,
                        device=input_device,
                        resolution=resolution,
                        image_token_budget=budget,
                        max_new_tokens=max_new_tokens,
                        no_think=no_think,
                    )
                    row["index"] = index
                    details_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    details_handle.flush()
                    results.append(row)
                    print(
                        f"{row['task']}[{index}] res={resolution} budget={budget} "
                        f"full={row['full_prediction']} compact={row['compact_prediction']} "
                        f"gold={row['answer_letter'] or row['answers']}"
                    )

    summary = summarize_vlm_results(
        results,
        target_compact_vs_full_accuracy=target_compact_vs_full,
    )
    summary["model"] = model_name
    summary["checkpoint"] = args.checkpoint or benchmark.get("checkpoint", "")
    summary["details_file"] = str(details_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if fail_under_target and not summary.get("target_passed", False):
        raise SystemExit(
            "Compact cache did not reach the configured compact-vs-full "
            f"target of {target_compact_vs_full:.3f}"
        )


if __name__ == "__main__":
    main()
