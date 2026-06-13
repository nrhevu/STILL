#!/usr/bin/env python3
"""Write a Markdown report from an evaluate_niah.py summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="NIAH Evaluation Report")
    parser.add_argument(
        "--target-relative-accuracy",
        type=float,
        default=0.95,
        help="Compact/full success-ratio target to report when full-cache metrics exist.",
    )
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    lines: list[str] = [f"# {args.title}", ""]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Model: `{summary.get('model', 'unknown')}`",
            f"- Checkpoint: `{summary.get('checkpoint') or 'none'}`",
            f"- Records: {summary.get('records', 0)}",
            f"- Compact success rate: {fmt(summary.get('overall_success_rate'))}",
            f"- Compact exact rate: {fmt(summary.get('overall_exact_rate'))}",
            f"- Mean compression: {fmt(summary.get('overall_mean_compression'))}x",
        ]
    )
    if "overall_full_success_rate" in summary:
        relative = float(summary.get("overall_relative_success_to_full", 0.0))
        target = float(args.target_relative_accuracy)
        status = "PASS" if relative >= target else "FAIL"
        lines.extend(
            [
                f"- Full-cache success rate: {fmt(summary.get('overall_full_success_rate'))}",
                f"- Compact/full success ratio: {relative:.4f} ({status}, target {target:.4f})",
            ]
        )
    for note in args.note:
        lines.append(f"- Note: {note}")

    rows = summary.get("by_context_depth", [])
    if rows:
        has_full = any("full_success_rate" in row for row in rows)
        has_task = any("task" in row for row in rows)
        lines.extend(["", "## By Context And Depth", ""])
        if has_full:
            if has_task:
                lines.append(
                    "| Context | Task | Depth % | Trials | Compact | Full | Compact/Full | "
                    "Compression | Decode s |"
                )
                lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            else:
                lines.append(
                    "| Context | Depth % | Trials | Compact | Full | Compact/Full | "
                    "Compression | Decode s |"
                )
                lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in rows:
                if has_task:
                    template = (
                        "| {context_length} | {task} | {depth_percent} | {trials} | "
                        "{success_rate} | {full_success_rate} | {relative_success_to_full} | "
                        "{mean_compression} | {mean_elapsed_seconds} |"
                    )
                else:
                    template = (
                        "| {context_length} | {depth_percent} | {trials} | {success_rate} | "
                        "{full_success_rate} | {relative_success_to_full} | {mean_compression} | "
                        "{mean_elapsed_seconds} |"
                    )
                lines.append(
                    template.format(
                        context_length=row.get("context_length"),
                        task=row.get("task", ""),
                        depth_percent=fmt(row.get("depth_percent")),
                        trials=row.get("trials"),
                        success_rate=fmt(row.get("success_rate")),
                        full_success_rate=fmt(row.get("full_success_rate")),
                        relative_success_to_full=fmt(row.get("relative_success_to_full")),
                        mean_compression=fmt(row.get("mean_compression")),
                        mean_elapsed_seconds=fmt(row.get("mean_elapsed_seconds")),
                    )
                )
        else:
            if has_task:
                lines.append(
                    "| Context | Task | Depth % | Trials | Success | Exact | "
                    "Compression | Decode s |"
                )
                lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            else:
                lines.append(
                    "| Context | Depth % | Trials | Success | Exact | Compression | Decode s |"
                )
                lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in rows:
                if has_task:
                    template = (
                        "| {context_length} | {task} | {depth_percent} | {trials} | "
                        "{success_rate} | {exact_rate} | {mean_compression} | "
                        "{mean_elapsed_seconds} |"
                    )
                else:
                    template = (
                        "| {context_length} | {depth_percent} | {trials} | {success_rate} | "
                        "{exact_rate} | {mean_compression} | {mean_elapsed_seconds} |"
                    )
                lines.append(
                    template.format(
                        context_length=row.get("context_length"),
                        task=row.get("task", ""),
                        depth_percent=fmt(row.get("depth_percent")),
                        trials=row.get("trials"),
                        success_rate=fmt(row.get("success_rate")),
                        exact_rate=fmt(row.get("exact_rate")),
                        mean_compression=fmt(row.get("mean_compression")),
                        mean_elapsed_seconds=fmt(row.get("mean_elapsed_seconds")),
                    )
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
