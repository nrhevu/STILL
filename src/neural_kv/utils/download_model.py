#!/usr/bin/env python3
"""Download a Hugging Face model snapshot with project quota checks."""

from __future__ import annotations

import argparse

from neural_kv.utils.hf_cache import configure_hf_cache
from neural_kv.utils.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-storage", default="10TB")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[
            "*.json",
            "*.model",
            "*.safetensors",
            "*.safetensors.index.json",
            "*.txt",
            "tokenizer*",
            "merges.txt",
            "vocab.json",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()

    from huggingface_hub import snapshot_download

    before = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage before model download: {before.summary()}", flush=True)
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        resume_download=True,
    )
    after = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"downloaded {args.model} to {path}", flush=True)
    print(f"storage after model download: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
