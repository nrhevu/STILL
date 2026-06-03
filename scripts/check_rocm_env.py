#!/usr/bin/env python3
"""Validate that the active uv environment is using a ROCm-enabled Torch build."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def main() -> None:
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
