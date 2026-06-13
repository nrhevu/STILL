#!/usr/bin/env python3
"""Run the configured Qwen3-235B 256k KV-compression benchmark and report."""

from __future__ import annotations

import argparse
import grp
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

DEFAULT_CONFIG = "configs/qwen3_235b_256k_niah.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--runtime",
        choices=["direct", "rocm-docker"],
        default="direct",
        help="Run directly in the host uv environment or through scripts/rocm_docker_run.sh.",
    )
    parser.add_argument(
        "--skip-rocm-check",
        action="store_true",
        help="Run even if the local ROCm preflight cannot see HIP devices.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the configured Hugging Face model snapshot before evaluation.",
    )
    parser.add_argument("--hf-home", default="data/hf_cache")
    parser.add_argument("--hip-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config {path} did not contain a YAML mapping")
    return payload


def _readable_groups() -> list[str]:
    names: list[str] = []
    for gid in os.getgroups():
        try:
            names.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            names.append(str(gid))
    return sorted(set(names))


def rocm_status() -> dict[str, Any]:
    kfd_path = Path("/dev/kfd")
    render_nodes = sorted(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
    rocm_smi = shutil.which("rocm-smi")
    rocm_smi_gfx950_count: int | None = None
    rocm_smi_error = ""
    if rocm_smi:
        try:
            completed = subprocess.run(
                [rocm_smi, "--showproductname"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            rocm_smi_gfx950_count = completed.stdout.count("GFX Version:\t\tgfx950")
            rocm_smi_error = completed.stderr.strip()
        except Exception as exc:
            rocm_smi_error = repr(exc)
    return {
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "torch_version_hip": torch.version.hip,
        "torch_cuda_is_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "dev_kfd_exists": kfd_path.exists(),
        "dev_kfd_read_write": os.access(kfd_path, os.R_OK | os.W_OK)
        if kfd_path.exists()
        else False,
        "dev_dri_render_nodes": len(render_nodes),
        "dev_dri_render_read_write": sum(
            1 for node in render_nodes if os.access(node, os.R_OK | os.W_OK)
        ),
        "user_groups": _readable_groups(),
        "rocm_smi_path": rocm_smi or "",
        "rocm_smi_gfx950_count": rocm_smi_gfx950_count,
        "rocm_smi_error": rocm_smi_error,
        "gpu_names": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }


def rocm_ready(status: dict[str, Any]) -> bool:
    return bool(
        status["torch_version_hip"]
        and status["torch_version_cuda"] is None
        and status["torch_cuda_is_available"]
        and int(status["torch_cuda_device_count"]) > 0
        and status["dev_kfd_read_write"]
    )


def output_paths(config: dict[str, Any]) -> dict[str, str]:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("config missing outputs mapping")
    required = {"records", "summary", "report"}
    missing = required - set(outputs)
    if missing:
        raise ValueError(f"config outputs missing keys: {sorted(missing)}")
    return {key: str(outputs[key]) for key in required}


def benchmark_command(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    dataset = config.get("dataset")
    compact_eval = config.get("compact_eval")
    if not isinstance(dataset, dict) or not isinstance(compact_eval, dict):
        raise ValueError("config must contain dataset and compact_eval mappings")
    outputs = output_paths(config)
    command = [
        sys.executable,
        "scripts/evaluate_niah.py",
        "--model",
        str(config["model"]),
        "--context-lengths",
        str(config["context_length"]),
        "--case-context-token-margin",
        str(dataset.get("case_context_token_margin", 0)),
        "--depths",
        str(dataset.get("depths", "0,25,50,75,100")),
        "--task",
        str(dataset.get("task", "mixed")),
        "--answer-mode",
        str(dataset.get("answer_mode", "open")),
        "--trials",
        str(dataset.get("trials", 1)),
        "--num-latents",
        str(compact_eval.get("num_latents", 1)),
        "--sink-tokens",
        str(compact_eval.get("sink_tokens", 0)),
        "--exact-tokens",
        str(compact_eval.get("exact_tokens", 0)),
        "--exact-strategy",
        str(compact_eval.get("exact_strategy", "lexical")),
        "--beta-base",
        str(compact_eval.get("beta_base", "zero")),
        "--device",
        "cuda",
        "--device-map",
        str(config.get("device_map", "auto")),
        "--max-memory",
        str(config.get("max_memory", "")),
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        str(config.get("attn_implementation", "sdpa")),
        "--rope-scaling",
        str(config.get("rope_scaling_json", "")),
        "--max-position-embeddings",
        str(config.get("max_position_embeddings", config["context_length"])),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prefill-chunk-size",
        str(config.get("prefill_chunk_size", 0)),
        "--output",
        outputs["records"],
        "--summary-output",
        outputs["summary"],
        "--max-storage",
        args.max_storage,
    ]
    if compact_eval.get("untrained_compactor", False):
        command.append("--untrained-compactor")
    if compact_eval.get("compare_full_cache", False):
        command.append("--compare-full-cache")
    return command


def report_command(config: dict[str, Any]) -> list[str]:
    outputs = output_paths(config)
    return [
        sys.executable,
        "scripts/write_niah_report.py",
        "--summary",
        outputs["summary"],
        "--output",
        outputs["report"],
        "--title",
        "Qwen3-235B 256k NIAH KV Compression Report",
        "--target-relative-accuracy",
        str(config.get("training_gate", {}).get("min_relative_accuracy_to_full", 0.95)),
        "--note",
        "256k uses empirical YaRN factor 8; official model-card example validates 131072.",
        "--note",
        (
            "This result uses an untrained num_latents=0 sink plus lexical exact-token "
            "baseline, not a trained latent compactor checkpoint."
        ),
    ]


def download_command(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/download_model.py",
        "--model",
        str(config["model"]),
        "--max-storage",
        args.max_storage,
    ]


def runtime_command(command: list[str], *, runtime: str) -> list[str]:
    if runtime == "direct":
        return command
    if runtime == "rocm-docker":
        return ["scripts/rocm_docker_run.sh", "python", *command[1:]]
    raise ValueError(f"unsupported runtime: {runtime}")


def runtime_ready(status: dict[str, Any], *, runtime: str) -> bool:
    if runtime == "direct":
        return rocm_ready(status)
    if runtime == "rocm-docker":
        return bool(
            Path("scripts/rocm_docker_run.sh").exists()
            and shutil.which("docker")
            and status["dev_kfd_exists"]
            and int(status["dev_dri_render_nodes"]) > 0
        )
    raise ValueError(f"unsupported runtime: {runtime}")


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_blocked_report(
    *,
    config: dict[str, Any],
    status: dict[str, Any],
    reason: str,
    benchmark: list[str],
    report: list[str],
) -> Path:
    outputs = output_paths(config)
    path = Path(outputs["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen3-235B 256k NIAH KV Compression Report",
        "",
        "## Status",
        "",
        f"Blocked at {datetime.now(UTC).isoformat()}.",
        "",
        f"Reason: {reason}",
        "",
        "## ROCm Preflight",
        "",
        "```json",
        json.dumps(status, indent=2, sort_keys=True),
        "```",
        "",
        "## Pending Benchmark Command",
        "",
        "```bash",
        command_text(benchmark),
        "```",
        "",
        "## Pending Report Command",
        "",
        "```bash",
        command_text(report),
        "```",
        "",
        "The >95% compact-vs-full target is not verified until the benchmark summary exists and "
        "reports `overall_relative_success_to_full >= 0.95` with a strong full-cache baseline.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(command: list[str], *, env: dict[str, str]) -> None:
    print(command_text(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bench_cmd = benchmark_command(config, args)
    report_cmd = report_command(config)
    dl_cmd = download_command(config, args)
    status = rocm_status()

    env = os.environ.copy()
    env.setdefault("HF_HOME", args.hf_home)
    env["HIP_VISIBLE_DEVICES"] = args.hip_visible_devices

    runtime_bench_cmd = runtime_command(bench_cmd, runtime=args.runtime)

    if args.dry_run:
        print("ROCm status:")
        print(json.dumps(status, indent=2, sort_keys=True))
        print(f"runtime: {args.runtime}")
        if args.download_model:
            print("download command:")
            print(command_text(dl_cmd))
        print("benchmark command:")
        print(command_text(runtime_bench_cmd))
        print("report command:")
        print(command_text(report_cmd))
        return

    if not args.skip_rocm_check and not runtime_ready(status, runtime=args.runtime):
        reason = (
            f"ROCm/HIP preflight failed for runtime={args.runtime}; "
            "this process cannot run the 8-GPU benchmark."
        )
        path = write_blocked_report(
            config=config,
            status=status,
            reason=reason,
            benchmark=runtime_bench_cmd,
            report=report_cmd,
        )
        raise SystemExit(f"{reason} Wrote blocked report: {path}")

    if args.download_model:
        run(dl_cmd, env=env)
    run(runtime_bench_cmd, env=env)
    run(report_cmd, env=env)


if __name__ == "__main__":
    main()
