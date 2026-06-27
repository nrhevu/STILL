#!/usr/bin/env python3
"""Prepare local JSONL VLM MCQ datasets with image files."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_kv.data.vlm import (
    normalize_mmmu_row,
    normalize_scienceqa_row,
    write_vlm_jsonl,
)
from neural_kv.utils.hf_cache import configure_hf_cache
from neural_kv.utils.storage import check_storage_quota, default_storage_roots

MMMU_SUBJECTS = (
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["scienceqa", "mmmu"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Optional per-output split row limit for smoke data.",
    )
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def _append_with_limit(rows: list[dict], row: dict | None, *, limit: int) -> bool:
    if row is None:
        return True
    if limit and len(rows) >= limit:
        return False
    rows.append(row)
    return True


def prepare_scienceqa(*, output_dir: Path, limit_per_split: int) -> None:
    from datasets import load_dataset

    dataset = load_dataset("derek-thomas/ScienceQA")
    for split in ("train", "validation", "test"):
        rows: list[dict] = []
        for index, source_row in enumerate(dataset[split]):
            keep_going = _append_with_limit(
                rows,
                normalize_scienceqa_row(
                    dict(source_row),
                    split=split,
                    row_index=index,
                    output_dir=output_dir,
                ),
                limit=limit_per_split,
            )
            if not keep_going:
                break
        count = write_vlm_jsonl(output_dir / f"{split}.jsonl", rows)
        print(f"wrote {count} ScienceQA {split} rows")


def prepare_mmmu(*, output_dir: Path, limit_per_split: int) -> None:
    from datasets import load_dataset

    split_rows: dict[str, list[dict]] = {"dev": [], "validation": []}
    for subject in MMMU_SUBJECTS:
        for split in ("dev", "validation"):
            dataset = load_dataset("MMMU/MMMU", subject, split=split)
            for index, source_row in enumerate(dataset):
                keep_going = _append_with_limit(
                    split_rows[split],
                    normalize_mmmu_row(
                        dict(source_row),
                        split=split,
                        subject=subject,
                        row_index=index,
                        output_dir=output_dir,
                    ),
                    limit=limit_per_split,
                )
                if not keep_going:
                    break
    for split, rows in split_rows.items():
        count = write_vlm_jsonl(output_dir / f"{split}.jsonl", rows)
        print(f"wrote {count} MMMU {split} rows")


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    storage_before = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage before vlm data prep: {storage_before}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "scienceqa":
        prepare_scienceqa(output_dir=output_dir, limit_per_split=args.limit_per_split)
    else:
        prepare_mmmu(output_dir=output_dir, limit_per_split=args.limit_per_split)

    storage_after = check_storage_quota(default_storage_roots(), args.max_storage).summary()
    print(f"storage after vlm data prep: {storage_after}")


if __name__ == "__main__":
    main()
