#!/usr/bin/env python3
"""Run VLM compactor evaluation, 95% target gate, and accepted-result archive."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

from neural_kv.utils.config import load_config
from neural_kv.utils.rocm import select_idle_gpu


def _csv_arg(value: str | list[int] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment/qwen3_vl_compactor_bench.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--archive-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--resolutions", default="")
    parser.add_argument("--image-token-budgets", default="")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--target-compact-vs-full", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device-map-auto", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-wait-for-gpu", action="store_true")
    parser.add_argument("--no-group-gate", action="store_true")
    parser.add_argument("--init-random-compactor", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _output_dir(args: argparse.Namespace, cfg: dict[str, object]) -> Path:
    return Path(args.output_dir or str(cfg.get("output_dir", "outputs/qwen3_vl_compactor_bench")))




def _required_packages() -> tuple[str, ...]:
    return ("torch", "torchvision", "PIL", "transformers", "datasets")


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def preflight_checks(args: argparse.Namespace) -> list[tuple[str, bool, str]]:
    """Return non-mutating checks needed before a real VLM acceptance run."""
    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    checks: list[tuple[str, bool, str]] = []

    if args.init_random_compactor:
        checks.append(("checkpoint", True, "using --init-random-compactor"))
    else:
        checkpoint = Path(args.checkpoint)
        checks.append(("checkpoint", checkpoint.exists(), str(checkpoint)))
    checks.append(("config", Path(args.config).exists(), str(args.config)))
    for package in _required_packages():
        checks.append((f"package:{package}", _package_available(package), package))

    output_dir = _output_dir(args, cfg)
    checks.append(("output_parent", output_dir.parent.exists(), str(output_dir.parent)))
    archive_dir = Path(
        args.archive_dir
        or str(cfg.get("benchmark", {}).get("archive_dir", "reports/vlm_compactor"))
    )
    checks.append(("archive_parent", archive_dir.parent.exists(), str(archive_dir.parent)))

    require_idle = bool(runtime.get("require_idle_gpu", False))
    device = args.device or str(runtime.get("model_load_device", "auto"))
    device_map_auto = bool(runtime.get("device_map_auto", False))
    if args.device_map_auto is not None:
        device_map_auto = bool(args.device_map_auto)
    needs_gpu = require_idle and device != "cpu" and (
        device in {"auto", "cuda"} or device.startswith("cuda") or device_map_auto
    )
    if needs_gpu:
        selected = select_idle_gpu(
            preferred=int(runtime.get("preferred_gpu", 7)),
            require_zero=True,
        )
        checks.append(("idle_gpu", selected is not None, str(selected)))
    return checks


def print_preflight(checks: list[tuple[str, bool, str]]) -> bool:
    failed = False
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}: {detail}", flush=True)
        failed = failed or not passed
    return not failed


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    cfg = load_config(args.config)
    benchmark = cfg.get("benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    runtime = cfg.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    model = cfg.get("model", {})
    if not isinstance(model, dict):
        model = {}

    output_dir = _output_dir(args, cfg)
    summary_path = output_dir / "summary.json"
    details_path = output_dir / "details.jsonl"
    archive_dir = Path(
        args.archive_dir or str(benchmark.get("archive_dir", "reports/vlm_compactor"))
    )
    target = float(
        args.target_compact_vs_full
        or benchmark.get("target_compact_vs_full_accuracy", 0.95)
    )
    limit = int(args.limit or benchmark.get("limit_per_dataset", 16))
    min_rows = int(args.min_rows or limit or 1)
    resolutions = _csv_arg(args.resolutions or benchmark.get("resolutions", ""))
    budgets = _csv_arg(args.image_token_budgets or benchmark.get("image_token_budgets", ""))
    max_new_tokens = int(args.max_new_tokens or benchmark.get("max_new_tokens", 32))
    device = args.device or str(runtime.get("model_load_device", "auto"))
    dtype = args.dtype or str(runtime.get("dtype", "bfloat16"))
    base_model = args.base_model or str(model.get("name", ""))
    device_map_auto = bool(runtime.get("device_map_auto", False))
    if args.device_map_auto is not None:
        device_map_auto = bool(args.device_map_auto)

    evaluate = [
        sys.executable,
        "scripts/evaluate_vlm_compactor.py",
        "--config",
        str(args.config),
        *( ["--checkpoint", str(args.checkpoint)] if args.checkpoint else [] ),
        "--output-dir",
        str(output_dir),
        "--summary-file",
        str(summary_path),
        "--details-file",
        str(details_path),
        "--limit",
        str(limit),
        "--max-new-tokens",
        str(max_new_tokens),
        "--target-compact-vs-full",
        str(target),
        "--fail-under-target",
        "--device",
        str(device),
        "--dtype",
        str(dtype),
    ]
    if args.init_random_compactor:
        evaluate.append("--init-random-compactor")
    if base_model:
        evaluate.extend(["--base-model", base_model])
    if resolutions:
        evaluate.extend(["--resolutions", resolutions])
    if budgets:
        evaluate.extend(["--image-token-budgets", budgets])
    if device_map_auto:
        evaluate.append("--device-map-auto")
    if args.no_wait_for_gpu:
        evaluate.append("--no-wait-for-gpu")

    check = [
        sys.executable,
        "scripts/check_vlm_compactor_target.py",
        str(summary_path),
        "--min-compact-vs-full-accuracy",
        str(target),
        "--min-rows",
        str(min_rows),
    ]
    if args.no_group_gate:
        check.append("--no-group-gate")

    archive = [
        sys.executable,
        "scripts/archive_vlm_compactor_result.py",
        str(summary_path),
        "--details-file",
        str(details_path),
        "--archive-dir",
        str(archive_dir),
        "--min-compact-vs-full-accuracy",
        str(target),
    ]
    if args.run_name:
        archive.extend(["--run-name", str(args.run_name)])
    if args.no_group_gate:
        archive.append("--no-group-gate")

    return [evaluate, check, archive]


def main() -> None:
    args = parse_args()
    if not args.skip_preflight:
        preflight_ok = print_preflight(preflight_checks(args))
        if args.preflight_only:
            if not preflight_ok:
                raise SystemExit(1)
            return
        if not preflight_ok:
            raise SystemExit("VLM acceptance preflight failed")

    output_dir = _output_dir(args, load_config(args.config))
    output_dir.mkdir(parents=True, exist_ok=True)
    for command in build_commands(args):
        print("+ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
