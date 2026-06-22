#!/usr/bin/env python3
"""Summarize streamed evaluate_checkpoint JSONL details."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-file", required=True)
    parser.add_argument("--summary-file", required=True)
    return parser.parse_args()


def read_details(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def summarize_details(rows: list[dict[str, object]]) -> dict[str, object]:
    compact_counts: collections.Counter[str] = collections.Counter()
    task_correct: collections.Counter[str] = collections.Counter()
    task_total: collections.Counter[str] = collections.Counter()
    compact_correct = 0
    full_correct = 0
    no_context_correct = 0
    compression_sum = 0.0
    for row in rows:
        gold = str(row["gold"])
        compact = str(row.get("compact") or "?")
        full = str(row.get("full") or "?")
        no_context = str(row.get("no_context") or "?")
        task = str(row.get("task") or "unknown")
        compact_correct += int(compact == gold)
        full_correct += int(full == gold)
        no_context_correct += int(no_context == gold)
        compact_counts[compact] += 1
        task_correct[task] += int(compact == gold)
        task_total[task] += 1
        compression_sum += float(row.get("compression") or 0.0)
    return {
        "rows": len(rows),
        "compact_accuracy": compact_correct / len(rows) if rows else 0.0,
        "full_accuracy": full_correct / len(rows) if rows else 0.0,
        "no_context_accuracy": no_context_correct / len(rows) if rows else 0.0,
        "mean_compression": compression_sum / len(rows) if rows else 0.0,
        "compact_prediction_counts": dict(sorted(compact_counts.items())),
        "task_accuracy": {
            task: task_correct[task] / task_total[task]
            for task in sorted(task_total)
        },
        "task_counts": dict(sorted(task_total.items())),
    }


def main() -> None:
    args = parse_args()
    summary = summarize_details(read_details(Path(args.details_file)))
    payload = json.dumps(summary, indent=2, sort_keys=True)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
