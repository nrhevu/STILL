"""Check VLM compactor compact-vs-full acceptance targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MIN_COMPACT_VS_FULL_ACCURACY = 0.95
DEFAULT_MIN_ROWS = 1

Check = tuple[str, bool, str]


def _float_metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _int_metric(payload: dict[str, Any], name: str) -> int | None:
    value = payload.get(name)
    return int(value) if isinstance(value, int) else None


def _target_metric(payload: dict[str, Any]) -> float | None:
    value = _float_metric(payload, "compact_full_agreement")
    if value is not None:
        return value
    value = _float_metric(payload, "compact_vs_full_accuracy")
    if value is not None:
        return value
    compact_accuracy = _float_metric(payload, "compact_accuracy")
    full_accuracy = _float_metric(payload, "full_accuracy")
    if compact_accuracy is None or full_accuracy is None or full_accuracy <= 0:
        return None
    return compact_accuracy / full_accuracy


def _group_name(group: dict[str, Any]) -> str:
    return (
        f"{group.get('task', 'unknown')}:"
        f"resolution={group.get('resolution')}:"
        f"image_token_budget={group.get('image_token_budget')}"
    )


def check_vlm_target(
    payload: dict[str, Any],
    *,
    min_compact_vs_full_accuracy: float = DEFAULT_MIN_COMPACT_VS_FULL_ACCURACY,
    min_rows: int = DEFAULT_MIN_ROWS,
    require_group_targets: bool = True,
) -> list[Check]:
    """Return acceptance checks for compact accuracy relative to full-cache accuracy."""
    rows = _int_metric(payload, "count") or _int_metric(payload, "rows")
    target_metric = _target_metric(payload)
    full_accuracy = _float_metric(payload, "full_accuracy")
    compact_accuracy = _float_metric(payload, "compact_accuracy")
    full_reference_count = _int_metric(payload, "full_reference_count")
    checks: list[Check] = [
        (
            "rows",
            rows is not None and rows >= min_rows,
            f"{rows} >= {min_rows}",
        ),
        (
            "full_reference_rows",
            (full_reference_count if full_reference_count is not None else rows or 0)
            >= min_rows,
            f"{full_reference_count if full_reference_count is not None else rows} >= {min_rows}",
        ),
        (
            "compact_full_agreement",
            target_metric is not None
            and target_metric >= min_compact_vs_full_accuracy,
            f"{target_metric} >= {min_compact_vs_full_accuracy} "
            f"(compact={compact_accuracy}, full={full_accuracy})",
        ),
    ]

    groups = payload.get("groups")
    if require_group_targets:
        if not isinstance(groups, list) or not groups:
            checks.append(("groups", False, "summary lacks non-empty groups"))
        else:
            failing_groups: list[str] = []
            for group in groups:
                if not isinstance(group, dict):
                    failing_groups.append("<invalid group>")
                    continue
                group_metric = _target_metric(group)
                if group_metric is None or group_metric < min_compact_vs_full_accuracy:
                    failing_groups.append(f"{_group_name(group)}={group_metric}")
            checks.append(
                (
                    "group_compact_full_agreement",
                    not failing_groups,
                    (
                        f"all groups >= {min_compact_vs_full_accuracy}"
                        if not failing_groups
                        else "failing: " + ", ".join(failing_groups)
                    ),
                )
            )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to evaluate_vlm_compactor.py summary JSON")
    parser.add_argument(
        "--min-compact-vs-full-accuracy",
        type=float,
        default=DEFAULT_MIN_COMPACT_VS_FULL_ACCURACY,
    )
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument(
        "--no-group-gate",
        action="store_true",
        help="Only check aggregate compact-vs-full accuracy, not each sweep group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    checks = check_vlm_target(
        payload,
        min_compact_vs_full_accuracy=args.min_compact_vs_full_accuracy,
        min_rows=args.min_rows,
        require_group_targets=not args.no_group_gate,
    )
    failed = False
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}: {detail}")
        failed = failed or not passed
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
