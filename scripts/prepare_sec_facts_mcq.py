#!/usr/bin/env python3
"""Build financial long-context MCQs from SEC companyfacts JSON."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from neural_kv.data import MCQExample, stable_id
from neural_kv.storage import check_storage_quota, default_storage_roots

DEFAULT_COMPANIES = {
    "Apple Inc.": "0000320193",
    "Microsoft Corp.": "0000789019",
    "Alphabet Inc.": "0001652044",
    "Amazon.com Inc.": "0001018724",
    "NVIDIA Corp.": "0001045810",
    "Tesla Inc.": "0001318605",
    "Meta Platforms Inc.": "0001326801",
    "Netflix Inc.": "0001065280",
    "Intel Corp.": "0000050863",
    "Advanced Micro Devices Inc.": "0000002488",
}


@dataclass(frozen=True)
class Fact:
    company: str
    concept: str
    label: str
    unit: str
    fiscal_year: int
    fiscal_period: str
    form: str
    filed: str
    value: str

    def line(self) -> str:
        return (
            f"Company: {self.company} | Concept: {self.label} ({self.concept}) | "
            f"Fiscal year: {self.fiscal_year} | Period: {self.fiscal_period} | "
            f"Unit: {self.unit} | Form: {self.form} | Filed: {self.filed} | Value: {self.value}"
        )

    def question(self) -> str:
        return (
            f"For {self.company}, concept {self.label} ({self.concept}), fiscal year "
            f"{self.fiscal_year}, period {self.fiscal_period}, unit {self.unit}, "
            "what is the reported value?"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/sec_facts")
    parser.add_argument(
        "--raw-dir",
        default="",
        help="Directory for SEC companyfacts JSON; defaults to OUTPUT_DIR/raw.",
    )
    parser.add_argument("--context-chars", type=int, default=50000)
    parser.add_argument("--train-rows", type=int, default=1200)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--target-placement",
        choices=["front", "random", "random_visible"],
        default="front",
        help="Where to place the target fact line within the packed context.",
    )
    parser.add_argument(
        "--visible-target-chars",
        type=int,
        default=22000,
        help="For random_visible placement, keep the target before this character offset.",
    )
    parser.add_argument("--max-storage", default="10TB")
    parser.add_argument(
        "--user-agent",
        default="neural-kv-compressor research contact@example.com",
        help="SEC requires a descriptive User-Agent.",
    )
    return parser.parse_args()


def _download_companyfacts(*, cik: str, raw_dir: Path, user_agent: str) -> dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"CIK{cik}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    path.write_text(payload, encoding="utf-8")
    time.sleep(0.2)
    return json.loads(payload)


def _extract_facts(company: str, payload: dict[str, object]) -> list[Fact]:
    facts: list[Fact] = []
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    if not isinstance(us_gaap, dict):
        return facts
    for concept, concept_payload in us_gaap.items():
        if not isinstance(concept_payload, dict):
            continue
        label = str(concept_payload.get("label") or concept)
        units = concept_payload.get("units", {})
        if not isinstance(units, dict):
            continue
        for unit, entries in units.items():
            if unit not in {"USD", "shares", "USD/shares"}:
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                fy = entry.get("fy")
                fp = str(entry.get("fp") or "")
                form = str(entry.get("form") or "")
                filed = str(entry.get("filed") or "")
                value = entry.get("val")
                if not isinstance(fy, int) or fp not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
                    continue
                if form not in {"10-K", "10-Q"} or value is None:
                    continue
                facts.append(
                    Fact(
                        company=company,
                        concept=str(concept),
                        label=label,
                        unit=str(unit),
                        fiscal_year=fy,
                        fiscal_period=fp,
                        form=form,
                        filed=filed,
                        value=str(value),
                    )
                )
    return facts


def _pack_document(
    target: Fact,
    facts: list[Fact],
    rng: random.Random,
    context_chars: int,
    *,
    target_placement: str,
    visible_target_chars: int,
) -> tuple[str, int]:
    target_line = target.line()
    filler_lines: list[str] = []
    shuffled = facts[:]
    rng.shuffle(shuffled)
    for fact in shuffled:
        line = fact.line()
        if line == target_line or line in filler_lines:
            continue
        current_length = len(target_line) + 1 + sum(len(item) + 1 for item in filler_lines)
        if current_length + len(line) > context_chars:
            continue
        filler_lines.append(line)
        packed_length = len(target_line) + 1 + sum(len(item) + 1 for item in filler_lines)
        if packed_length >= context_chars * 0.9:
            break
    if target_placement == "front":
        target_index = 0
    elif target_placement == "random":
        target_index = rng.randrange(len(filler_lines) + 1)
    elif target_placement == "random_visible":
        if visible_target_chars <= 0:
            raise ValueError("visible_target_chars must be positive")
        eligible_indices = [0]
        prefix_length = 0
        for index, line in enumerate(filler_lines, start=1):
            prefix_length += len(line) + 1
            if prefix_length > visible_target_chars:
                break
            eligible_indices.append(index)
        target_index = rng.choice(eligible_indices)
    else:
        raise ValueError(f"Unsupported target placement: {target_placement}")
    lines = filler_lines[:]
    lines.insert(target_index, target_line)
    return "\n".join(lines)[:context_chars], target_index


def _make_options(
    answer: str,
    values: list[str],
    document: str,
    rng: random.Random,
) -> list[str] | None:
    choices = [answer]
    seen = {answer}
    for candidate in rng.sample(values, k=min(len(values), 2048)):
        if candidate in seen or candidate in document:
            continue
        choices.append(candidate)
        seen.add(candidate)
        if len(choices) == 4:
            break
    if len(choices) != 4:
        return None
    rng.shuffle(choices)
    return choices


def _build_rows(
    *,
    split: str,
    source_facts: list[Fact],
    all_facts: list[Fact],
    count: int,
    context_chars: int,
    seed: int,
    target_placement: str,
    visible_target_chars: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    candidates = source_facts[:]
    rng.shuffle(candidates)
    values = [fact.value for fact in all_facts]
    rows: list[dict[str, object]] = []
    for fact in candidates:
        document, target_line_index = _pack_document(
            fact,
            all_facts,
            rng,
            context_chars,
            target_placement=target_placement,
            visible_target_chars=visible_target_chars,
        )
        choices = _make_options(fact.value, values, document, rng)
        if choices is None:
            continue
        example = MCQExample(
            id=stable_id("sec", split, fact.company, fact.concept, fact.filed, fact.value),
            split=split,
            source="sec_companyfacts",
            context=document,
            question=fact.question(),
            choices=choices,
            answer_index=choices.index(fact.value),
            answer=fact.value,
        )
        payload = asdict(example)
        payload["target_placement"] = target_placement
        payload["target_line_index"] = target_line_index
        payload["target_char_offset"] = document.find(fact.line())
        if target_placement == "random_visible":
            payload["visible_target_chars"] = visible_target_chars
        rows.append(payload)
        if len(rows) >= count:
            break
    return rows


def _write_rows(path: Path, rows: list[dict[str, object] | MCQExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row) if is_dataclass(row) else row
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    before = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage before SEC download: {before.summary()}", flush=True)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "raw"
    facts: list[Fact] = []
    for company, cik in DEFAULT_COMPANIES.items():
        payload = _download_companyfacts(cik=cik, raw_dir=raw_dir, user_agent=args.user_agent)
        facts.extend(_extract_facts(company, payload))
    facts = [fact for fact in facts if 2015 <= fact.fiscal_year <= 2025]
    facts.sort(
        key=lambda fact: (
            fact.company,
            fact.concept,
            fact.fiscal_year,
            fact.fiscal_period,
            fact.filed,
        )
    )
    rng = random.Random(args.seed)
    rng.shuffle(facts)
    train_cut = int(len(facts) * 0.75)
    validation_cut = int(len(facts) * 0.875)
    train_facts = facts[:train_cut]
    validation_facts = facts[train_cut:validation_cut]
    test_facts = facts[validation_cut:]
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _build_rows(
        split="train",
        source_facts=train_facts,
        all_facts=facts,
        count=args.train_rows,
        context_chars=args.context_chars,
        seed=args.seed,
        target_placement=args.target_placement,
        visible_target_chars=args.visible_target_chars,
    )
    validation = _build_rows(
        split="validation",
        source_facts=validation_facts,
        all_facts=facts,
        count=args.validation_rows,
        context_chars=args.context_chars,
        seed=args.seed + 1,
        target_placement=args.target_placement,
        visible_target_chars=args.visible_target_chars,
    )
    test = _build_rows(
        split="test",
        source_facts=test_facts,
        all_facts=facts,
        count=args.test_rows,
        context_chars=args.context_chars,
        seed=args.seed + 2,
        target_placement=args.target_placement,
        visible_target_chars=args.visible_target_chars,
    )
    print(
        f"facts={len(facts)} wrote train={_write_rows(output_dir / 'train.jsonl', train)} "
        f"validation={_write_rows(output_dir / 'validation.jsonl', validation)} "
        f"test={_write_rows(output_dir / 'test.jsonl', test)}",
        flush=True,
    )
    after = check_storage_quota(default_storage_roots(), args.max_storage)
    print(f"storage after SEC download: {after.summary()}", flush=True)


if __name__ == "__main__":
    main()
