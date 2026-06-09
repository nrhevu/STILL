"""Synthetic needle-in-a-haystack data helpers."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from neural_kv.data import stable_id


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


def make_niah_case(
    tokenizer,
    *,
    context_length: int,
    depth_percent: float,
    trial: int,
    seed: int,
) -> NiahCase:
    rng = random.Random(seed + context_length * 10_003 + int(depth_percent * 100) * 97 + trial)
    key = f"NIAH-{context_length}-{int(depth_percent * 100):04d}-{trial}-{rng.randrange(1000, 9999)}"
    value = f"VX{rng.randrange(10**9, 10**10 - 1)}Z"
    needle = (
        f"\nIMPORTANT NEEDLE RECORD: The secret value for retrieval key {key} is {value}. "
        "When asked for this key, answer with only the secret value.\n"
    )
    needle_tokens = token_ids(tokenizer, needle)
    if len(needle_tokens) >= context_length:
        raise ValueError(
            f"Needle has {len(needle_tokens)} tokens, which does not fit "
            f"context_length={context_length}"
        )
    filler_budget = context_length - len(needle_tokens)
    prefix_tokens = int(round(filler_budget * (depth_percent / 100.0)))
    suffix_tokens = filler_budget - prefix_tokens
    prefix = filler_text_for_tokens(tokenizer, token_count=prefix_tokens, rng=rng)
    suffix = filler_text_for_tokens(tokenizer, token_count=suffix_tokens, rng=rng)
    context = prefix + needle + suffix
    actual_context_tokens = len(token_ids(tokenizer, context))
    return NiahCase(
        context_length=context_length,
        depth_percent=depth_percent,
        trial=trial,
        key=key,
        value=value,
        context=context,
        actual_context_tokens=actual_context_tokens,
        needle_char_offset=context.find(needle.strip()),
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
    rng = random.Random(seed + case.context_length * 31 + int(case.depth_percent * 100) + case.trial)
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
        "id": stable_id("niah", split, case.key, str(case.trial), str(case.depth_percent)),
        "split": split,
        "source": "synthetic_niah",
        "context": case.context,
        "question": niah_question(case),
        "choices": choices,
        "answer_index": choices.index(case.value),
        "answer": case.value,
        "niah_key": case.key,
        "depth_percent": case.depth_percent,
        "trial": case.trial,
        "actual_context_tokens": case.actual_context_tokens,
        "needle_char_offset": case.needle_char_offset,
    }
    return row


def case_payload(case: NiahCase) -> dict[str, object]:
    return asdict(case)
