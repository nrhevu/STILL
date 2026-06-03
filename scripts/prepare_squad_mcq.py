#!/usr/bin/env python3
"""Build long-context MCQ rows from SQuAD exact-answer examples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from neural_kv.data import MCQExample, stable_id, write_jsonl
from neural_kv.storage import check_storage_quota, default_storage_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/squad_mcq")
    parser.add_argument("--context-chars", type=int, default=50000)
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-storage", default="10TB")
    return parser.parse_args()


def _load_squad_rows(split: str) -> list[dict[str, object]]:
    rows = []
    for row in load_dataset("squad", split=split):
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            continue
        answer = str(answers[0]).strip()
        question = str(row["question"]).strip()
        context = str(row["context"]).strip()
        if len(answer) < 2 or not question or answer.lower() not in context.lower():
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "title": str(row.get("title") or ""),
                "question": question,
                "answer": answer,
                "context": context,
            }
        )
    return rows


def _pack_context(
    *,
    target: dict[str, object],
    pool: list[dict[str, object]],
    rng: random.Random,
    context_chars: int,
) -> str:
    passages = [str(target["context"])]
    shuffled = pool[:]
    rng.shuffle(shuffled)
    for row in shuffled:
        passage = str(row["context"])
        if passage in passages:
            continue
        if sum(len(item) + 2 for item in passages) + len(passage) > context_chars:
            continue
        passages.append(passage)
        if sum(len(item) + 2 for item in passages) >= context_chars * 0.85:
            break
    rng.shuffle(passages)
    return "\n\n".join(passages)[:context_chars]


def _options(
    *,
    answer: str,
    answer_pool: list[str],
    document: str,
    rng: random.Random,
) -> list[str] | None:
    document_lower = document.lower()
    choices = [answer]
    seen = {answer.lower()}
    for candidate in rng.sample(answer_pool, k=min(len(answer_pool), 1024)):
        key = candidate.lower()
        if key in seen or key in document_lower:
            continue
        choices.append(candidate)
        seen.add(key)
        if len(choices) == 4:
            break
    if len(choices) != 4:
        return None
    rng.shuffle(choices)
    return choices


def _build_split(
    *,
    split_name: str,
    source_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    count: int,
    context_chars: int,
    seed: int,
) -> list[MCQExample]:
    rng = random.Random(seed)
    candidates = source_rows[:]
    rng.shuffle(candidates)
    answer_pool = [str(row["answer"]) for row in all_rows]
    examples: list[MCQExample] = []
    for row in candidates:
        document = _pack_context(
            target=row,
            pool=all_rows,
            rng=rng,
            context_chars=context_chars,
        )
        choices = _options(
            answer=str(row["answer"]),
            answer_pool=answer_pool,
            document=document,
            rng=rng,
        )
        if choices is None:
            continue
        answer_index = choices.index(str(row["answer"]))
        examples.append(
            MCQExample(
                id=stable_id("squad", split_name, str(row["id"])),
                split=split_name,
                source="squad",
                context=document,
                question=str(row["question"]),
                choices=choices,
                answer_index=answer_index,
                answer=str(row["answer"]),
            )
        )
        if len(examples) >= count:
            break
    return examples


def main() -> None:
    args = parse_args()
    before = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage before SQuAD download: {before.summary()}", flush=True)
    train_rows = _load_squad_rows("train")
    validation_source = _load_squad_rows("validation")
    midpoint = len(validation_source) // 2
    all_rows = [*train_rows, *validation_source]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _build_split(
        split_name="train",
        source_rows=train_rows,
        all_rows=all_rows,
        count=args.train_rows,
        context_chars=args.context_chars,
        seed=args.seed,
    )
    validation = _build_split(
        split_name="validation",
        source_rows=validation_source[:midpoint],
        all_rows=all_rows,
        count=args.validation_rows,
        context_chars=args.context_chars,
        seed=args.seed + 1,
    )
    test = _build_split(
        split_name="test",
        source_rows=validation_source[midpoint:],
        all_rows=all_rows,
        count=args.test_rows,
        context_chars=args.context_chars,
        seed=args.seed + 2,
    )
    print(
        f"wrote train={write_jsonl(output_dir / 'train.jsonl', train)} "
        f"validation={write_jsonl(output_dir / 'validation.jsonl', validation)} "
        f"test={write_jsonl(output_dir / 'test.jsonl', test)}",
        flush=True,
    )
    after = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage after SQuAD download: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
