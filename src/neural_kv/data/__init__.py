"""Dataset schemas and data preparation helpers."""

from neural_kv.data.mcq import (
    MCQExample,
    answer_letter,
    build_mcq_examples,
    chunk_texts,
    download_gutenberg_texts,
    format_mcq_prompt,
    load_hf_texts,
    normalize_text,
    read_jsonl,
    stable_id,
    strip_gutenberg_boilerplate,
    write_jsonl,
)
from neural_kv.data.ruler import DEFAULT_RULER_TASKS, build_ruler_mcq_examples

__all__ = [
    "DEFAULT_RULER_TASKS",
    "MCQExample",
    "answer_letter",
    "build_mcq_examples",
    "build_ruler_mcq_examples",
    "chunk_texts",
    "download_gutenberg_texts",
    "format_mcq_prompt",
    "load_hf_texts",
    "normalize_text",
    "read_jsonl",
    "stable_id",
    "strip_gutenberg_boilerplate",
    "write_jsonl",
]
