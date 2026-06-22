#!/usr/bin/env python3
"""Check RULER 200k Qwen3-235B 8x acceptance metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MIN_COMPACT_ACCURACY = 0.95
DEFAULT_MIN_COMPRESSION = 8.0
DEFAULT_MIN_FULL_ACCURACY = 0.98
DEFAULT_MIN_TASK_ACCURACY = 0.95
DEFAULT_MIN_ROWS = 64
DEFAULT_REQUIRED_TASKS = (
    "common_words_extraction",
    "niah_multikey",
    "niah_multivalue",
    "niah_single",
    "qa",
    "variable_tracking",
)

Check = tuple[str, bool, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to an evaluate_checkpoint.py summary JSON file")
    parser.add_argument("--min-compact-accuracy", type=float, default=DEFAULT_MIN_COMPACT_ACCURACY)
    parser.add_argument("--min-compression", type=float, default=DEFAULT_MIN_COMPRESSION)
    parser.add_argument("--min-full-accuracy", type=float, default=DEFAULT_MIN_FULL_ACCURACY)
    parser.add_argument("--min-task-accuracy", type=float, default=DEFAULT_MIN_TASK_ACCURACY)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument(
        "--required-tasks",
        default=",".join(DEFAULT_REQUIRED_TASKS),
        help="Comma-separated task names that must appear in task_counts.",
    )
    parser.add_argument(
        "--coverage-summary",
        default="",
        help="Optional check_ruler_coverage.py summary JSON that must have zero failures.",
    )
    parser.add_argument(
        "--no-task-gate",
        action="store_true",
        help="Only check aggregate compact accuracy, not each task_accuracy entry.",
    )
    return parser.parse_args()


def _metric(payload: dict[str, object], name: str) -> float | None:
    value = payload.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _int_metric(payload: dict[str, object], name: str) -> int | None:
    value = payload.get(name)
    return int(value) if isinstance(value, int) else None


def _task_accuracy(payload: dict[str, object]) -> dict[str, float]:
    value = payload.get("task_accuracy")
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for task, accuracy in value.items():
        if isinstance(task, str) and isinstance(accuracy, (int, float)):
            result[task] = float(accuracy)
    return result


def _task_counts(payload: dict[str, object]) -> dict[str, int]:
    value = payload.get("task_counts")
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for task, count in value.items():
        if isinstance(task, str) and isinstance(count, int):
            result[task] = int(count)
    return result


def _parse_required_tasks(value: str) -> tuple[str, ...]:
    return tuple(task.strip() for task in value.split(",") if task.strip())


def _coverage_checks(
    coverage_payload: dict[str, object] | None,
    *,
    min_rows: int,
) -> list[Check]:
    if coverage_payload is None:
        return []
    failed = _int_metric(coverage_payload, "failed")
    rows = _int_metric(coverage_payload, "rows")
    return [
        (
            "coverage_failed",
            failed == 0,
            f"{failed} == 0",
        ),
        (
            "coverage_rows",
            rows is not None and rows >= min_rows,
            f"{rows} >= {min_rows}",
        ),
    ]


def check_ruler_target(
    payload: dict[str, object],
    *,
    min_compact_accuracy: float = DEFAULT_MIN_COMPACT_ACCURACY,
    min_compression: float = DEFAULT_MIN_COMPRESSION,
    min_full_accuracy: float = DEFAULT_MIN_FULL_ACCURACY,
    min_task_accuracy: float | None = DEFAULT_MIN_TASK_ACCURACY,
    min_rows: int = DEFAULT_MIN_ROWS,
    required_tasks: tuple[str, ...] = DEFAULT_REQUIRED_TASKS,
    coverage_payload: dict[str, object] | None = None,
) -> list[Check]:
    rows = _int_metric(payload, "rows")
    compact_accuracy = _metric(payload, "compact_accuracy")
    full_accuracy = _metric(payload, "full_accuracy")
    mean_compression = _metric(payload, "mean_compression") or _metric(payload, "compression")

    checks: list[Check] = [
        (
            "rows",
            rows is not None and rows >= min_rows,
            f"{rows} >= {min_rows}",
        ),
        (
            "compact_accuracy",
            compact_accuracy is not None and compact_accuracy >= min_compact_accuracy,
            f"{compact_accuracy} >= {min_compact_accuracy}",
        ),
        (
            "mean_compression",
            mean_compression is not None and mean_compression >= min_compression,
            f"{mean_compression} >= {min_compression}",
        ),
        (
            "full_accuracy",
            full_accuracy is not None and full_accuracy >= min_full_accuracy,
            f"{full_accuracy} >= {min_full_accuracy}",
        ),
    ]

    task_counts = _task_counts(payload)
    missing_tasks = [task for task in required_tasks if task_counts.get(task, 0) <= 0]
    if missing_tasks:
        checks.append(
            (
                "required_tasks",
                False,
                "missing: " + ", ".join(missing_tasks),
            )
        )
    else:
        checks.append(
            (
                "required_tasks",
                True,
                f"present: {', '.join(required_tasks)}",
            )
        )

    if min_task_accuracy is not None:
        task_accuracy = _task_accuracy(payload)
        failing = {
            task: accuracy
            for task, accuracy in sorted(task_accuracy.items())
            if accuracy < min_task_accuracy
        }
        if not task_accuracy:
            checks.append(("task_accuracy", False, "summary lacks task_accuracy"))
        elif failing:
            failures = ", ".join(f"{task}={accuracy}" for task, accuracy in failing.items())
            checks.append(
                (
                    "task_accuracy",
                    False,
                    f"all tasks >= {min_task_accuracy}; failing: {failures}",
                )
            )
        else:
            checks.append(
                (
                    "task_accuracy",
                    True,
                    f"{len(task_accuracy)} tasks >= {min_task_accuracy}",
                )
            )
    checks.extend(_coverage_checks(coverage_payload, min_rows=min_rows))
    return checks


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    coverage_payload = None
    if args.coverage_summary:
        coverage_payload = json.loads(Path(args.coverage_summary).read_text(encoding="utf-8"))
    checks = check_ruler_target(
        payload,
        min_compact_accuracy=args.min_compact_accuracy,
        min_compression=args.min_compression,
        min_full_accuracy=args.min_full_accuracy,
        min_task_accuracy=None if args.no_task_gate else args.min_task_accuracy,
        min_rows=args.min_rows,
        required_tasks=_parse_required_tasks(args.required_tasks),
        coverage_payload=coverage_payload,
    )

    failed = [name for name, passed, _ in checks if not passed]
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}: {detail}")
    if failed:
        raise SystemExit(f"RULER 200k target failed: {', '.join(failed)}")
    print("RULER 200k target passed")


if __name__ == "__main__":
    main()
