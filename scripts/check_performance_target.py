#!/usr/bin/env python3
"""Check a training summary against the Baseten STILL target metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to a train_still.py summary.json file")
    parser.add_argument("--min-compact-accuracy", type=float, default=0.85)
    parser.add_argument("--min-compression", type=float, default=8.0)
    parser.add_argument("--min-utilization", type=float, default=0.90)
    return parser.parse_args()


def _metric(payload: dict[str, object], name: str) -> float | None:
    value = payload.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    compact_accuracy = _metric(payload, "compact_accuracy")
    full_accuracy = _metric(payload, "full_accuracy")
    no_context_accuracy = _metric(payload, "no_context_accuracy")
    mean_compression = _metric(payload, "mean_compression") or _metric(payload, "compression")

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "compact_accuracy",
            compact_accuracy is not None and compact_accuracy >= args.min_compact_accuracy,
            f"{compact_accuracy} >= {args.min_compact_accuracy}",
        )
    )
    checks.append(
        (
            "mean_compression",
            mean_compression is not None and mean_compression >= args.min_compression,
            f"{mean_compression} >= {args.min_compression}",
        )
    )

    has_utilization_inputs = (
        no_context_accuracy is not None
        and full_accuracy is not None
        and compact_accuracy is not None
    )
    if has_utilization_inputs:
        denominator = full_accuracy - no_context_accuracy
        utilization = (
            (compact_accuracy - no_context_accuracy) / denominator
            if abs(denominator) > 1e-12
            else 0.0
        )
        payload["mcq_utilization"] = utilization
        checks.append(
            (
                "mcq_utilization",
                utilization >= args.min_utilization,
                f"{utilization} >= {args.min_utilization}",
            )
        )
    else:
        checks.append(
            (
                "mcq_utilization",
                False,
                "summary lacks no_context_accuracy/full_accuracy/compact_accuracy",
            )
        )

    failed = [name for name, passed, _ in checks if not passed]
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}: {detail}")
    if failed:
        raise SystemExit(f"performance target failed: {', '.join(failed)}")
    print("performance target passed")


if __name__ == "__main__":
    main()
