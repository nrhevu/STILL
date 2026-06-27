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
from neural_kv.data.vlm import (
    format_vlm_mcq_prompt,
    normalize_mmmu_row,
    normalize_scienceqa_row,
    parse_answer_index,
    parse_options,
    vlm_answer_letter,
)

__all__ = [
    "DEFAULT_RULER_TASKS",
    "MCQExample",
    "answer_letter",
    "build_mcq_examples",
    "build_ruler_mcq_examples",
    "chunk_texts",
    "download_gutenberg_texts",
    "format_mcq_prompt",
    "format_vlm_mcq_prompt",
    "load_hf_texts",
    "normalize_text",
    "normalize_mmmu_row",
    "normalize_scienceqa_row",
    "parse_answer_index",
    "parse_options",
    "read_jsonl",
    "stable_id",
    "strip_gutenberg_boilerplate",
    "vlm_answer_letter",
    "write_jsonl",
]
