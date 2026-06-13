"""Synthetic needle-in-a-haystack data helpers."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from neural_kv.data import stable_id

NIAH_TASKS = {"single", "multi_needle", "two_hop", "mixed"}


@dataclass(frozen=True)
class NiahCase:
    context_length: int
    depth_percent: float
    trial: int
    key: str
    value: str
    context: str
    actual_context_tokens: int
    needle_char_offset: int
    task: str = "single"
    secondary_key: str = ""
    record_count: int = 1


FILLER_TOPICS = [
    "quarterly ledger reconciliation",
    "warehouse climate inspection",
    "legacy invoice migration",
    "regional routing memo",
    "archival policy update",
    "supplier quality review",
    "call-center transcript digest",
    "maintenance ticket summary",
]


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def filler_text_for_tokens(tokenizer, *, token_count: int, rng: random.Random) -> str:
    if token_count <= 0:
        return ""
    lines: list[str] = []
    ids: list[int] = []
    line_index = 0
    while len(ids) < token_count:
        chunk: list[str] = []
        for _ in range(128):
            topic = rng.choice(FILLER_TOPICS)
            marker = rng.randrange(10_000_000, 99_999_999)
            control = rng.randrange(100_000, 999_999)
            chunk.append(
                f"Archive line {line_index:05d}: topic={topic}; marker={marker}; "
                f"control value CV{control}; this sentence is unrelated to any secret key.\n"
            )
            line_index += 1
        lines.extend(chunk)
        ids = token_ids(tokenizer, "".join(lines))
    return tokenizer.decode(ids[:token_count], skip_special_tokens=False)


def _resolve_task(task: str, *, trial: int, depth_percent: float) -> str:
    if task not in NIAH_TASKS:
        raise ValueError(f"task must be one of {sorted(NIAH_TASKS)}")
    if task != "mixed":
        return task
    tasks = ["single", "multi_needle", "two_hop"]
    return tasks[(trial + int(depth_percent * 10)) % len(tasks)]


def _context_with_records(
    tokenizer,
    *,
    records: list[tuple[str, float]],
    context_length: int,
    rng: random.Random,
) -> str:
    token_lengths = [len(token_ids(tokenizer, record)) for record, _ in records]
    record_tokens = sum(token_lengths)
    if record_tokens >= context_length:
        raise ValueError(
            f"Records have {record_tokens} tokens, which do not fit context_length={context_length}"
        )

    filler_budget = context_length - record_tokens
    placed = sorted(
        (
            (int(round(filler_budget * (max(0.0, min(depth, 100.0)) / 100.0))), record)
            for record, depth in records
        ),
        key=lambda item: item[0],
    )

    parts: list[str] = []
    used_filler = 0
    for target_offset, record in placed:
        gap = max(0, min(target_offset, filler_budget) - used_filler)
        parts.append(filler_text_for_tokens(tokenizer, token_count=gap, rng=rng))
        parts.append(record)
        used_filler += gap
    suffix = filler_budget - used_filler
    parts.append(filler_text_for_tokens(tokenizer, token_count=suffix, rng=rng))
    return "".join(parts)


def _single_records(*, key: str, value: str) -> list[tuple[str, str]]:
    needle = (
        f"\nIMPORTANT NEEDLE RECORD: The secret value for retrieval key {key} is {value}. "
        "When asked for this key, answer with only the secret value.\n"
    )
    return [(key, needle)]


def _multi_needle_records(
    *,
    rng: random.Random,
    key: str,
    value: str,
    distractors: int = 7,
) -> list[tuple[str, str]]:
    records = _single_records(key=key, value=value)
    seen_keys = {key}
    seen_values = {value}
    for idx in range(distractors):
        distractor_key = f"NIAH-DISTRACTOR-{idx}-{rng.randrange(1000, 9999)}"
        while distractor_key in seen_keys:
            distractor_key = f"NIAH-DISTRACTOR-{idx}-{rng.randrange(1000, 9999)}"
        distractor_value = f"VX{rng.randrange(10**9, 10**10 - 1)}Z"
        while distractor_value in seen_values:
            distractor_value = f"VX{rng.randrange(10**9, 10**10 - 1)}Z"
        seen_keys.add(distractor_key)
        seen_values.add(distractor_value)
        records.extend(_single_records(key=distractor_key, value=distractor_value))
    rng.shuffle(records)
    return records


def _two_hop_records(
    *,
    rng: random.Random,
    key: str,
    value: str,
) -> tuple[list[tuple[str, str]], str]:
    secondary_key = f"VAULT-{rng.randrange(10_000_000, 99_999_999)}"
    route = (
        f"\nROUTING NEEDLE RECORD: retrieval key {key} maps to vault key {secondary_key}. "
        f"To answer a question about {key}, resolve the vault key first.\n"
    )
    vault = (
        f"\nVAULT NEEDLE RECORD: the secret value stored under vault key {secondary_key} "
        f"is {value}. Use this vault record as the answer source.\n"
    )
    distractor_vault = f"VAULT-{rng.randrange(10_000_000, 99_999_999)}"
    distractor = (
        f"\nVAULT NEEDLE RECORD: the secret value stored under vault key {distractor_vault} "
        f"is VX{rng.randrange(10**9, 10**10 - 1)}Z. This record is unrelated.\n"
    )
    return [(key, route), (secondary_key, vault), (distractor_vault, distractor)], secondary_key


def make_niah_case(
    tokenizer,
    *,
    context_length: int,
    depth_percent: float,
    trial: int,
    seed: int,
    task: str = "single",
) -> NiahCase:
    resolved_task = _resolve_task(task, trial=trial, depth_percent=depth_percent)
    rng = random.Random(
        seed
        + context_length * 10_003
        + int(depth_percent * 100) * 97
        + trial
        + sum(ord(char) for char in resolved_task) * 13
    )
    key = (
        f"NIAH-{context_length}-{int(depth_percent * 100):04d}-{trial}-{rng.randrange(1000, 9999)}"
    )
    value = f"VX{rng.randrange(10**9, 10**10 - 1)}Z"
    secondary_key = ""

    if resolved_task == "single":
        records = _single_records(key=key, value=value)
        depth_records = [(records[0][1], depth_percent)]
    elif resolved_task == "multi_needle":
        records = _multi_needle_records(rng=rng, key=key, value=value)
        step = 100.0 / max(len(records), 1)
        depth_records = []
        for index, (record_key, record_text) in enumerate(records):
            depth = depth_percent if record_key == key else min(100.0, index * step)
            depth_records.append((record_text, depth))
    elif resolved_task == "two_hop":
        records, secondary_key = _two_hop_records(rng=rng, key=key, value=value)
        vault_depth = (depth_percent + 50.0) % 100.0
        depth_records = [
            (records[0][1], depth_percent),
            (records[1][1], vault_depth),
            (records[2][1], min(100.0, (vault_depth + 25.0) % 100.0)),
        ]
    else:
        raise AssertionError(f"unhandled task {resolved_task!r}")

    context = _context_with_records(
        tokenizer,
        records=depth_records,
        context_length=context_length,
        rng=rng,
    )
    actual_context_tokens = len(token_ids(tokenizer, context))
    needle_text = records[0][1].strip()
    if resolved_task == "two_hop":
        needle_text = records[1][1].strip()
    return NiahCase(
        context_length=context_length,
        depth_percent=depth_percent,
        trial=trial,
        key=key,
        value=value,
        context=context,
        actual_context_tokens=actual_context_tokens,
        needle_char_offset=context.find(needle_text),
        task=resolved_task,
        secondary_key=secondary_key,
        record_count=len(records),
    )


def normalize_niah_answer(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def niah_question(case: NiahCase) -> str:
    return f"What is the secret value for retrieval key {case.key}?"


def niah_user_prompt(case: NiahCase, *, no_think: bool = True) -> str:
    prefix = "/no_think\n" if no_think else ""
    return f"{prefix}Question: {niah_question(case)}\nAnswer with only the secret value."


def niah_case_to_mcq_row(
    case: NiahCase,
    *,
    split: str,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(
        seed + case.context_length * 31 + int(case.depth_percent * 100) + case.trial
    )
    choices = [case.value]
    seen = {case.value}
    while len(choices) < 4:
        candidate = f"VX{rng.randrange(10**9, 10**10 - 1)}Z"
        if candidate in seen or candidate in case.context:
            continue
        choices.append(candidate)
        seen.add(candidate)
    rng.shuffle(choices)
    row = {
        "id": stable_id(
            "niah", split, case.task, case.key, str(case.trial), str(case.depth_percent)
        ),
        "split": split,
        "source": f"synthetic_niah_{case.task}",
        "context": case.context,
        "question": niah_question(case),
        "choices": choices,
        "answer_index": choices.index(case.value),
        "answer": case.value,
        "niah_key": case.key,
        "niah_task": case.task,
        "secondary_key": case.secondary_key,
        "record_count": case.record_count,
        "depth_percent": case.depth_percent,
        "trial": case.trial,
        "actual_context_tokens": case.actual_context_tokens,
        "needle_char_offset": case.needle_char_offset,
    }
    return row


def case_payload(case: NiahCase) -> dict[str, object]:
    return asdict(case)
