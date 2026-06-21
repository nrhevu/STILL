#!/usr/bin/env python3
"""ROCm runtime helpers and environment validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

_USAGE_KEY_RE = re.compile(r"(?:use|util|busy).*%?", re.IGNORECASE)
_PERCENT_RE = re.compile(r"GPU\[(\d+)\].*?(\d+(?:\.\d+)?)%")


def _extract_usage_values(payload: object) -> dict[int, float]:
    values: dict[int, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            index_match = re.search(r"(?:card|gpu)\D*(\d+)", str(key), re.IGNORECASE)
            if index_match and isinstance(value, dict):
                gpu_index = int(index_match.group(1))
                for metric_key, metric_value in value.items():
                    if _USAGE_KEY_RE.search(str(metric_key)):
                        try:
                            values[gpu_index] = float(str(metric_value).strip("%"))
                        except ValueError:
                            pass
            values.update(_extract_usage_values(value))
    elif isinstance(payload, list):
        for value in payload:
            values.update(_extract_usage_values(value))
    return values


def gpu_utilization() -> dict[int, float]:
    """Return ROCm GPU utilization percentages keyed by physical GPU index."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        values = _extract_usage_values(parsed)
        if values:
            return values
    return {int(index): float(value) for index, value in _PERCENT_RE.findall(result.stdout)}


def select_idle_gpu(*, preferred: int = 7, require_zero: bool = True) -> int | None:
    """Select an idle GPU, preferring ``preferred`` when its utilization is zero."""
    usage = gpu_utilization()
    if not usage:
        return None
    if usage.get(preferred) == 0:
        return preferred
    idle = [index for index, value in sorted(usage.items()) if value == 0]
    if idle:
        return idle[0]
    if require_zero:
        return None
    return min(usage, key=usage.__getitem__)


def apply_visible_device_for_idle_gpu(
    *,
    preferred: int = 7,
    require_zero: bool = True,
) -> int | None:
    """Set HIP_VISIBLE_DEVICES to an idle GPU before Torch is imported."""
    selected = select_idle_gpu(preferred=preferred, require_zero=require_zero)
    if selected is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-utilization", action="store_true")
    parser.add_argument("--select-idle", action="store_true")
    parser.add_argument("--preferred-gpu", type=int, default=7)
    parser.add_argument("--allow-nonzero", action="store_true")
    args = parser.parse_args()

    if args.show_utilization:
        print(json.dumps(gpu_utilization(), indent=2, sort_keys=True))
    if args.select_idle:
        selected = apply_visible_device_for_idle_gpu(
            preferred=args.preferred_gpu,
            require_zero=not args.allow_nonzero,
        )
        if selected is None:
            raise SystemExit("No 0% utilization GPU is available")
        print(f"HIP_VISIBLE_DEVICES={selected}")

    import torch

    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.version.hip={torch.version.hip}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")

    failures: list[str] = []
    if torch.version.hip is None:
        failures.append("Torch is not a ROCm/HIP build.")
    if torch.version.cuda is not None:
        failures.append("Torch reports CUDA support; expected ROCm/HIP only.")
    if Path("/dev/kfd").exists() and not os.access("/dev/kfd", os.R_OK | os.W_OK):
        failures.append("Current process cannot read/write /dev/kfd.")
    if not torch.cuda.is_available():
        failures.append("ROCm Torch does not see an available HIP device via torch.cuda.")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    for idx in range(torch.cuda.device_count()):
        print(f"gpu[{idx}]={torch.cuda.get_device_name(idx)}")
    print("ROCm Torch environment is ready.")


if __name__ == "__main__":
    main()
