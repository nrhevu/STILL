#!/usr/bin/env python3
"""Download bounded public text data and build extractive MCQ JSONL splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from neural_kv.data import (
    build_mcq_examples,
    chunk_texts,
    download_gutenberg_texts,
    load_hf_texts,
    write_jsonl,
)
from neural_kv.storage import check_storage_quota, default_storage_roots

DEFAULT_GUTENBERG_IDS = [
    1342,   # Pride and Prejudice
    11,     # Alice's Adventures in Wonderland
    1661,   # Sherlock Holmes
    84,     # Frankenstein
    2701,   # Moby-Dick
    345,    # Dracula
    2600,   # War and Peace
    4300,   # Ulysses
    174,    # Dorian Gray
    1952,   # Yellow Wallpaper
    98,     # A Tale of Two Cities
    1400,   # Great Expectations
    844,    # Importance of Being Earnest
    1260,   # Jane Eyre
    768,    # Wuthering Heights
    46,     # A Christmas Carol
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["gutenberg", "hf"], default="gutenberg")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", default="data/mcq")
    parser.add_argument("--hf-cache-dir", default="data/hf_cache")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--gutenberg-ids",
        default=",".join(str(book_id) for book_id in DEFAULT_GUTENBERG_IDS),
    )
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--eval-docs", type=int, default=128)
    parser.add_argument("--questions-per-doc", type=int, default=2)
    parser.add_argument("--context-chars", type=int, default=12000)
    parser.add_argument("--chunks-per-text", type=int, default=1)
    parser.add_argument("--chunk-stride-chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    hf_cache_dir = Path(args.hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir / "datasets"))

    roots = default_storage_roots()
    before = check_storage_quota(roots, args.max_storage)
    print(f"storage before download: {before.summary()}", flush=True)

    if args.source == "gutenberg":
        book_ids = [int(value) for value in args.gutenberg_ids.split(",") if value.strip()]
        print(
            f"downloading/reusing {len(book_ids)} Gutenberg books into {args.raw_dir}",
            flush=True,
        )
        texts = download_gutenberg_texts(book_ids=book_ids, raw_dir=args.raw_dir)
        split_at = max(1, int(len(texts) * 0.75))
        train_texts = texts[:split_at]
        heldout_texts = texts[split_at:]
        heldout_midpoint = max(1, len(heldout_texts) // 2)
        validation_texts = heldout_texts[:heldout_midpoint]
        test_texts = heldout_texts[heldout_midpoint:] or heldout_texts[:heldout_midpoint]
        source_name = "project_gutenberg"
    else:
        print(f"loading Hugging Face dataset {args.dataset}/{args.dataset_config}", flush=True)
        train_texts = load_hf_texts(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split="train",
            text_field=args.text_field,
            max_rows=max(args.train_docs * 20, args.train_docs),
        )
        # Wikitext has a validation split. For other datasets, this name may need changing.
        eval_texts = load_hf_texts(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split="validation",
            text_field=args.text_field,
            max_rows=max(args.eval_docs * 20, args.eval_docs),
        )
        validation_texts = eval_texts[: max(1, len(eval_texts) // 2)]
        test_texts = eval_texts[max(1, len(eval_texts) // 2) :]
        source_name = f"{args.dataset}/{args.dataset_config}"

    train_texts = chunk_texts(
        train_texts,
        context_chars=args.context_chars,
        chunks_per_text=args.chunks_per_text,
        stride_chars=args.chunk_stride_chars,
    )
    validation_texts = chunk_texts(
        validation_texts,
        context_chars=args.context_chars,
        chunks_per_text=args.chunks_per_text,
        stride_chars=args.chunk_stride_chars,
    )
    test_texts = chunk_texts(
        test_texts,
        context_chars=args.context_chars,
        chunks_per_text=args.chunks_per_text,
        stride_chars=args.chunk_stride_chars,
    )

    train_rows = build_mcq_examples(
        texts=train_texts,
        split="train",
        source=source_name,
        max_docs=args.train_docs,
        questions_per_doc=args.questions_per_doc,
        context_chars=args.context_chars,
        seed=args.seed,
    )
    validation_rows = build_mcq_examples(
        texts=validation_texts,
        split="validation",
        source=source_name,
        max_docs=args.eval_docs // 2,
        questions_per_doc=args.questions_per_doc,
        context_chars=args.context_chars,
        seed=args.seed + 1,
    )
    test_rows = build_mcq_examples(
        texts=test_texts,
        split="test",
        source=source_name,
        max_docs=args.eval_docs // 2,
        questions_per_doc=args.questions_per_doc,
        context_chars=args.context_chars,
        seed=args.seed + 2,
    )

    train_count = write_jsonl(output_dir / "train.jsonl", train_rows)
    validation_count = write_jsonl(output_dir / "validation.jsonl", validation_rows)
    test_count = write_jsonl(output_dir / "test.jsonl", test_rows)

    after = check_storage_quota(roots, args.max_storage)
    print(f"wrote train={train_count} validation={validation_count} test={test_count}", flush=True)
    print(f"storage after download: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
