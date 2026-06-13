#!/usr/bin/env python3
"""Check a summary against the Baseten STILL target metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to a summary.json file")
    parser.add_argument("--min-compact-accuracy", type=float, default=0.85)
    parser.add_argument("--min-compression", type=float, default=8.0)
    parser.add_argument("--min-utilization", type=float, default=0.90)
    parser.add_argument(
        "--min-relative-accuracy",
        type=float,
        default=0.0,
        help="Optional compact_accuracy / full_accuracy floor, e.g. 0.95.",
    )
    return parser.parse_args()


def _metric(payload: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    compact_accuracy = _metric(payload, "compact_accuracy", "overall_success_rate")
    full_accuracy = _metric(payload, "full_accuracy", "overall_full_success_rate")
    no_context_accuracy = _metric(payload, "no_context_accuracy")
    mean_compression = _metric(
        payload, "mean_compression", "compression", "overall_mean_compression"
    )

    checks: list[tuple[str, bool | None, str]] = []
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
    if args.min_relative_accuracy > 0:
        relative_accuracy = _metric(
            payload,
            "relative_accuracy_to_full",
            "overall_relative_success_to_full",
        )
        if relative_accuracy is None and compact_accuracy is not None and full_accuracy is not None:
            relative_accuracy = compact_accuracy / full_accuracy if full_accuracy > 0 else None
        payload["relative_accuracy_to_full"] = relative_accuracy
        checks.append(
            (
                "relative_accuracy_to_full",
                relative_accuracy is not None and relative_accuracy >= args.min_relative_accuracy,
                f"{relative_accuracy} >= {args.min_relative_accuracy}",
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
        is_niah_summary = (
            "overall_success_rate" in payload or "overall_relative_success_to_full" in payload
        )
        checks.append(
            (
                "mcq_utilization",
                None if is_niah_summary else False,
                (
                    "not present for NIAH summary schema"
                    if is_niah_summary
                    else "summary lacks no_context_accuracy/full_accuracy/compact_accuracy"
                ),
            )
        )

    failed = [name for name, passed, _ in checks if passed is False]
    for name, passed, detail in checks:
        status = "SKIP" if passed is None else "PASS" if passed else "FAIL"
        print(f"{status} {name}: {detail}")
    if failed:
        raise SystemExit(f"performance target failed: {', '.join(failed)}")
    print("performance target passed")


if __name__ == "__main__":
    main()
