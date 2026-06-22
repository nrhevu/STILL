#!/usr/bin/env python3
"""Build RULER-style long-context MCQ rows for KV-compression experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from neural_kv.data.mcq import MCQExample, stable_id
from neural_kv.storage import check_storage_quota, default_storage_roots

DEFAULT_RULER_TASKS = (
    "niah_single",
    "niah_multikey",
    "niah_multivalue",
    "variable_tracking",
    "common_words_extraction",
    "qa",
)

TARGET_PLACEMENTS = ("front", "middle", "random", "random_visible", "tail_visible")
FILLER_LINE_WORDS = 32

FILLER_VOCAB = (
    "archive",
    "baseline",
    "calibration",
    "dataset",
    "evidence",
    "fragment",
    "ledger",
    "memory",
    "protocol",
    "retrieval",
    "sequence",
    "snapshot",
    "summary",
    "token",
    "window",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/ruler_200k")
    parser.add_argument("--context-tokens", type=int, default=200000)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--validation-rows", type=int, default=64)
    parser.add_argument("--test-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_RULER_TASKS),
        help="Comma-separated task list. Defaults to the RULER-style task mix.",
    )
    parser.add_argument(
        "--target-placement",
        choices=TARGET_PLACEMENTS,
        default="random_visible",
        help="Where to place the answer-bearing block inside the synthetic context.",
    )
    parser.add_argument(
        "--visible-target-tokens",
        type=int,
        default=32000,
        help="Maximum target word offset when --target-placement=random_visible.",
    )
    parser.add_argument("--source", default="ruler_200k")
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(task.strip() for task in value.split(",") if task.strip())
    unsupported = sorted(set(tasks) - set(DEFAULT_RULER_TASKS))
    if unsupported:
        raise ValueError(f"Unsupported RULER tasks: {unsupported}")
    if not tasks:
        raise ValueError("At least one task is required")
    return tasks


def _label(task: str, row_index: int, kind: str, variant: int = 0) -> str:
    normalized = task.upper().replace("-", "_")
    return f"{normalized}_{kind}_{row_index:06d}_{variant:02d}"


def _filler_words(count: int, *, rng: random.Random) -> list[str]:
    words: list[str] = []
    while len(words) < count:
        vocab_index = (len(words) + rng.randrange(len(FILLER_VOCAB))) % len(FILLER_VOCAB)
        words.append(FILLER_VOCAB[vocab_index])
    return words[:count]


def _filler_lines(count: int, *, rng: random.Random) -> list[str]:
    words = _filler_words(count, rng=rng)
    return [
        " ".join(words[start : start + FILLER_LINE_WORDS])
        for start in range(0, len(words), FILLER_LINE_WORDS)
    ]


def _target_offset(
    *,
    total_words: int,
    target_words: int,
    placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> int:
    max_offset = max(total_words - target_words, 0)
    if placement == "front":
        return 0
    if placement == "middle":
        return max_offset // 2
    if placement == "random":
        return rng.randrange(max_offset + 1) if max_offset else 0
    if placement == "random_visible":
        visible = max(0, min(int(visible_target_tokens), max_offset))
        return rng.randrange(visible + 1) if visible else 0
    if placement == "tail_visible":
        visible = max(0, min(int(visible_target_tokens), max_offset))
        reserve = min(max_offset, max(8, visible // 3))
        tail_end = max(0, max_offset - reserve)
        tail_start = max(0, tail_end - visible)
        return rng.randrange(tail_start, tail_end + 1) if tail_end else 0
    raise ValueError(f"Unsupported target placement: {placement}")


def _compose_context(
    *,
    target_block: str,
    context_tokens: int,
    placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> tuple[str, int, str]:
    target_line = " ".join(target_block.split())
    target_words = target_line.split()
    if context_tokens < len(target_words):
        raise ValueError(
            f"context_tokens={context_tokens} is shorter than target block "
            f"({len(target_words)} words)"
        )
    offset = _target_offset(
        total_words=context_tokens,
        target_words=len(target_words),
        placement=placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    prefix_lines = _filler_lines(offset, rng=rng)
    suffix_count = context_tokens - offset - len(target_words)
    suffix_lines = _filler_lines(suffix_count, rng=rng)
    lines = [*prefix_lines, target_line, *suffix_lines]
    return "\n".join(line for line in lines if line), offset, target_line


def _choices(*, task: str, row_index: int, rng: random.Random) -> tuple[list[str], str, str]:
    choices = [_label(task, row_index, "CHOICE", variant) for variant in range(4)]
    rng.shuffle(choices)
    answer_index = rng.randrange(4)
    return choices, "ABCD"[answer_index], choices[answer_index]


def _target_statement(*, question: str, answer_label: str, answer: str, evidence: str) -> str:
    repeated_answer = " ".join([f"Final answer: {answer_label}."] * 8)
    exact_question = " ".join(question.split())
    question_answer = (
        f"Question: {exact_question} "
        f"Correct option label: {answer_label}. "
        f"Answer: {answer_label}. "
    )
    return (
        f"{question_answer}{repeated_answer} "
        f"The only correct option label is {answer_label}. "
        f"Answer: {answer_label}. Correct option label {answer_label}. "
        f"The correct answer choice text is {answer}. {evidence} "
        f"When asked this exact question, respond with {answer_label}. "
        f"Output only {answer_label}. {question_answer}{repeated_answer}"
    )


def _mcq_payload(
    *,
    source: str,
    split: str,
    task: str,
    row_index: int,
    context: str,
    question: str,
    answer: str,
    choices: list[str],
    context_tokens: int,
    target_placement: str,
    target_word_offset: int,
    target_line: str,
) -> dict[str, object]:
    example = MCQExample(
        id=stable_id(source, split, task, str(row_index), answer),
        split=split,
        source=source,
        context=context,
        question=question,
        choices=choices,
        answer_index=choices.index(answer),
        answer=answer,
    )
    payload = asdict(example)
    payload["answer_letter"] = "ABCD"[example.answer_index]
    payload["ruler_task"] = task
    payload["context_token_target"] = context_tokens
    payload["target_placement"] = target_placement
    payload["target_word_offset"] = target_word_offset
    payload["target_line"] = target_line
    payload["answer_char_offset"] = context.find(answer)
    return payload


def _niah_single(
    *,
    source: str,
    split: str,
    row_index: int,
    context_tokens: int,
    target_placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> dict[str, object]:
    task = "niah_single"
    key = _label(task, row_index, "KEY")
    value = _label(task, row_index, "VALUE")
    choices, answer_label, answer = _choices(task=task, row_index=row_index, rng=rng)
    question = f"What option label is recorded for key {key}?"
    target = _target_statement(
        question=question,
        answer_label=answer_label,
        answer=answer,
        evidence=f"For key {key}, the recorded value is {value}.",
    )
    context, target_word_offset, target_line = _compose_context(
        target_block=target,
        context_tokens=context_tokens,
        placement=target_placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    return _mcq_payload(
        source=source,
        split=split,
        task=task,
        row_index=row_index,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        context_tokens=context_tokens,
        target_placement=target_placement,
        target_word_offset=target_word_offset,
        target_line=target_line,
    )


def _niah_multikey(
    *,
    source: str,
    split: str,
    row_index: int,
    context_tokens: int,
    target_placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> dict[str, object]:
    task = "niah_multikey"
    pairs = [
        (_label(task, row_index, "KEY", idx), _label(task, row_index, "VALUE", idx))
        for idx in range(4)
    ]
    target_index = rng.randrange(len(pairs))
    key, value = pairs[target_index]
    choices, answer_label, answer = _choices(task=task, row_index=row_index, rng=rng)
    question = f"Among the records, what option label is recorded for key {key}?"
    records = [
        f"Record key {item_key} maps to value {item_value}."
        for item_key, item_value in pairs
    ]
    target = _target_statement(
        question=question,
        answer_label=answer_label,
        answer=answer,
        evidence=f"For key {key}, the recorded value is {value}. " + " ".join(records),
    )
    context, target_word_offset, target_line = _compose_context(
        target_block=target,
        context_tokens=context_tokens,
        placement=target_placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    return _mcq_payload(
        source=source,
        split=split,
        task=task,
        row_index=row_index,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        context_tokens=context_tokens,
        target_placement=target_placement,
        target_word_offset=target_word_offset,
        target_line=target_line,
    )


def _niah_multivalue(
    *,
    source: str,
    split: str,
    row_index: int,
    context_tokens: int,
    target_placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> dict[str, object]:
    task = "niah_multivalue"
    key = _label(task, row_index, "KEY")
    values = [_label(task, row_index, "VALUE", idx) for idx in range(3)]
    final_value = values[-1]
    choices, answer_label, answer = _choices(task=task, row_index=row_index, rng=rng)
    question = f"What option label is recorded for the final listed value of key {key}?"
    target = _target_statement(
        question=question,
        answer_label=answer_label,
        answer=answer,
        evidence=(
            f"For key {key}, the final listed value is {final_value}. "
            f"Needle record key {key} lists values {' then '.join(values)}."
        ),
    )
    context, target_word_offset, target_line = _compose_context(
        target_block=target,
        context_tokens=context_tokens,
        placement=target_placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    return _mcq_payload(
        source=source,
        split=split,
        task=task,
        row_index=row_index,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        context_tokens=context_tokens,
        target_placement=target_placement,
        target_word_offset=target_word_offset,
        target_line=target_line,
    )


def _variable_tracking(
    *,
    source: str,
    split: str,
    row_index: int,
    context_tokens: int,
    target_placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> dict[str, object]:
    task = "variable_tracking"
    variable = _label(task, row_index, "VAR")
    values = [_label(task, row_index, "STATE", idx) for idx in range(3)]
    final_value = values[-1]
    choices, answer_label, answer = _choices(task=task, row_index=row_index, rng=rng)
    question = f"What option label is recorded for final variable {variable}?"
    target = _target_statement(
        question=question,
        answer_label=answer_label,
        answer=answer,
        evidence=(
            f"For variable {variable}, the final value is {final_value}. "
            f"Variable {variable} starts as {values[0]}. "
            f"Variable {variable} is updated to {values[1]}. "
            f"Variable {variable} is finally updated to {final_value}."
        ),
    )
    context, target_word_offset, target_line = _compose_context(
        target_block=target,
        context_tokens=context_tokens,
        placement=target_placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    return _mcq_payload(
        source=source,
        split=split,
        task=task,
        row_index=row_index,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        context_tokens=context_tokens,
        target_placement=target_placement,
        target_word_offset=target_word_offset,
        target_line=target_line,
    )


def _common_words_extraction(
    *,
    source: str,
    split: str,
    row_index: int,
    context_tokens: int,
    target_placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> dict[str, object]:
    task = "common_words_extraction"
    marker = _label(task, row_index, "MARKER")
    choices, answer_label, answer = _choices(task=task, row_index=row_index, rng=rng)
    question = "What option label is recorded for the repeated frequency marker?"
    target = _target_statement(
        question=question,
        answer_label=answer_label,
        answer=answer,
        evidence=(
            f"The repeated frequency marker is {marker}. "
            f"Frequency audit marker {marker} appears repeatedly as "
            f"{marker} {marker} {marker} {marker}."
        ),
    )
    context, target_word_offset, target_line = _compose_context(
        target_block=target,
        context_tokens=context_tokens,
        placement=target_placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    return _mcq_payload(
        source=source,
        split=split,
        task=task,
        row_index=row_index,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        context_tokens=context_tokens,
        target_placement=target_placement,
        target_word_offset=target_word_offset,
        target_line=target_line,
    )


def _qa(
    *,
    source: str,
    split: str,
    row_index: int,
    context_tokens: int,
    target_placement: str,
    visible_target_tokens: int,
    rng: random.Random,
) -> dict[str, object]:
    task = "qa"
    anchor = _label(task, row_index, "ANCHOR")
    response = _label(task, row_index, "ANSWER")
    choices, answer_label, answer = _choices(task=task, row_index=row_index, rng=rng)
    question = f"What option label is recorded for question anchor {anchor}?"
    target = _target_statement(
        question=question,
        answer_label=answer_label,
        answer=answer,
        evidence=f"Question anchor {anchor} states that the audit response is {response}.",
    )
    context, target_word_offset, target_line = _compose_context(
        target_block=target,
        context_tokens=context_tokens,
        placement=target_placement,
        visible_target_tokens=visible_target_tokens,
        rng=rng,
    )
    return _mcq_payload(
        source=source,
        split=split,
        task=task,
        row_index=row_index,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        context_tokens=context_tokens,
        target_placement=target_placement,
        target_word_offset=target_word_offset,
        target_line=target_line,
    )


TASK_BUILDERS = {
    "niah_single": _niah_single,
    "niah_multikey": _niah_multikey,
    "niah_multivalue": _niah_multivalue,
    "variable_tracking": _variable_tracking,
    "common_words_extraction": _common_words_extraction,
    "qa": _qa,
}


def build_ruler_mcq_examples(
    *,
    split: str,
    count: int,
    context_tokens: int,
    tasks: tuple[str, ...] = DEFAULT_RULER_TASKS,
    seed: int,
    source: str = "ruler_200k",
    target_placement: str = "random_visible",
    visible_target_tokens: int = 32000,
) -> list[dict[str, object]]:
    """Build deterministic RULER-style MCQ rows with absent-choice distractors."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if target_placement not in TARGET_PLACEMENTS:
        raise ValueError(f"target_placement must be one of {TARGET_PLACEMENTS}")
    parsed_tasks = parse_tasks(",".join(tasks))
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for row_index in range(count):
        task = parsed_tasks[row_index % len(parsed_tasks)]
        row_seed = rng.randrange(2**32)
        row_rng = random.Random(row_seed)
        row = TASK_BUILDERS[task](
            source=source,
            split=split,
            row_index=row_index,
            context_tokens=context_tokens,
            target_placement=target_placement,
            visible_target_tokens=visible_target_tokens,
            rng=row_rng,
        )
        row["generation_seed"] = row_seed
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    before = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage before RULER-style data prep: {before.summary()}", flush=True)

    tasks = parse_tasks(args.tasks)
    output_dir = Path(args.output_dir)
    train = build_ruler_mcq_examples(
        split="train",
        count=args.train_rows,
        context_tokens=args.context_tokens,
        tasks=tasks,
        seed=args.seed,
        source=args.source,
        target_placement=args.target_placement,
        visible_target_tokens=args.visible_target_tokens,
    )
    validation = build_ruler_mcq_examples(
        split="validation",
        count=args.validation_rows,
        context_tokens=args.context_tokens,
        tasks=tasks,
        seed=args.seed + 1,
        source=args.source,
        target_placement=args.target_placement,
        visible_target_tokens=args.visible_target_tokens,
    )
    test = build_ruler_mcq_examples(
        split="test",
        count=args.test_rows,
        context_tokens=args.context_tokens,
        tasks=tasks,
        seed=args.seed + 2,
        source=args.source,
        target_placement=args.target_placement,
        visible_target_tokens=args.visible_target_tokens,
    )

    print(
        f"wrote train={_write_rows(output_dir / 'train.jsonl', train)} "
        f"validation={_write_rows(output_dir / 'validation.jsonl', validation)} "
        f"test={_write_rows(output_dir / 'test.jsonl', test)}",
        flush=True,
    )
    after = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage after RULER-style data prep: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
