"""Public-corpus MCQ data preparation utilities."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z][A-Za-z0-9_-]{2,}\b")
_NUMBER_RE = re.compile(r"\b(?:\d{4}|\d+(?:\.\d+)?%?)\b")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{5,}\b")


@dataclass(frozen=True)
class MCQExample:
    """One extractive multiple-choice row."""

    id: str
    split: str
    source: str
    context: str
    question: str
    choices: list[str]
    answer_index: int
    answer: str


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_texts(
    texts: list[str],
    *,
    context_chars: int,
    chunks_per_text: int,
    stride_chars: int,
) -> list[str]:
    """Expand long source texts into overlapping fixed-character chunks."""
    if chunks_per_text <= 1:
        return texts
    stride = stride_chars if stride_chars > 0 else max(context_chars // 2, 1)
    chunks: list[str] = []
    for text in texts:
        for chunk_idx in range(chunks_per_text):
            start = chunk_idx * stride
            end = start + context_chars
            if start >= len(text):
                break
            chunk = text[start:end]
            if len(chunk) >= min(400, context_chars):
                chunks.append(chunk)
    return chunks


def _candidate_answers(text: str) -> list[str]:
    candidates: list[str] = []
    for regex in (_NUMBER_RE, _CAPITALIZED_RE, _WORD_RE):
        for match in regex.finditer(text):
            value = match.group(0).strip(".,;:!?()[]{}\"'")
            if 3 <= len(value) <= 64:
                candidates.append(value)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _question_for_answer(context: str, answer: str) -> str:
    return (
        "Which option is explicitly mentioned in the cached document? "
        "Only one option appears in the document."
    )


def _sample_absent_distractors(
    *,
    answer: str,
    all_answers: list[str],
    context_lower: str,
    rng: random.Random,
    count: int = 3,
) -> list[str]:
    distractors: list[str] = []
    seen = {answer.lower()}
    max_attempts = min(len(all_answers), 512)
    for candidate in rng.sample(all_answers, k=max_attempts):
        key = candidate.lower()
        if key in seen or key in context_lower:
            continue
        distractors.append(candidate)
        seen.add(key)
        if len(distractors) == count:
            break
    return distractors


def build_mcq_examples(
    *,
    texts: Iterable[str],
    split: str,
    source: str,
    max_docs: int,
    questions_per_doc: int,
    context_chars: int,
    seed: int,
) -> list[MCQExample]:
    """Build deterministic extractive MCQs from raw document texts."""
    rng = random.Random(seed)
    examples: list[MCQExample] = []
    all_answers: list[str] = []
    normalized_texts = [normalize_text(text)[:context_chars] for text in texts]
    normalized_texts = [text for text in normalized_texts if len(text) >= 400]

    for text in normalized_texts:
        all_answers.extend(_candidate_answers(text))
    all_answers = list(dict.fromkeys(all_answers))

    for doc_idx, context in enumerate(normalized_texts[:max_docs]):
        candidates = _candidate_answers(context)
        if len(candidates) < 4:
            continue
        rng.shuffle(candidates)
        made = 0
        context_lower = context.lower()
        for answer in candidates:
            distractors = _sample_absent_distractors(
                answer=answer,
                all_answers=all_answers,
                context_lower=context_lower,
                rng=rng,
            )
            if len(distractors) < 3:
                continue
            choices = [answer, *distractors]
            rng.shuffle(choices)
            answer_index = choices.index(answer)
            question = _question_for_answer(context, answer)
            examples.append(
                MCQExample(
                    id=stable_id(source, split, str(doc_idx), answer, question),
                    split=split,
                    source=source,
                    context=context,
                    question=question,
                    choices=choices,
                    answer_index=answer_index,
                    answer=answer,
                )
            )
            made += 1
            if made >= questions_per_doc:
                break
    return examples


def write_jsonl(path: str | Path, rows: Iterable[MCQExample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_hf_texts(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    max_rows: int,
) -> list[str]:
    """Load text rows from Hugging Face Datasets lazily."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts: list[str] = []
    for row in dataset:
        value = row.get(text_field)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        if len(texts) >= max_rows:
            break
    return texts


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove common Project Gutenberg header/footer markers when present."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    upper = text.upper()
    start = 0
    for marker in start_markers:
        index = upper.find(marker)
        if index >= 0:
            newline = text.find("\n", index)
            start = newline + 1 if newline >= 0 else index
            break
    end = len(text)
    for marker in end_markers:
        index = upper.find(marker)
        if index >= 0:
            end = index
            break
    return text[start:end].strip()


def download_gutenberg_texts(
    *,
    book_ids: list[int],
    raw_dir: str | Path,
    timeout_seconds: int = 30,
) -> list[str]:
    """Download public-domain Gutenberg books into ``raw_dir`` and return texts."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    for book_id in book_ids:
        destination = raw_path / f"gutenberg_{book_id}.txt"
        if destination.exists() and destination.stat().st_size > 0:
            text = destination.read_text(encoding="utf-8", errors="ignore")
            texts.append(strip_gutenberg_boilerplate(text))
            continue
        candidates = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        ]
        last_error: Exception | None = None
        for url in candidates:
            try:
                with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                    payload = response.read()
                text = payload.decode("utf-8", errors="ignore")
                destination.write_text(text, encoding="utf-8")
                texts.append(strip_gutenberg_boilerplate(text))
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(
                f"Failed to download Gutenberg book {book_id}: {last_error}"
            ) from last_error
    return texts


def format_mcq_prompt(row: dict[str, object]) -> str:
    choices = row["choices"]
    if not isinstance(choices, list):
        raise ValueError("choices must be a list")
    labels = ["A", "B", "C", "D"]
    rendered = "\n".join(f"{labels[idx]}. {choice}" for idx, choice in enumerate(choices))
    return (
        "/no_think\n"
        "Use the previous document context to answer the question.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{rendered}\n\n"
        "Answer with only the single capital letter of the correct option.\n"
        "Answer:"
    )


def answer_letter(row: dict[str, object]) -> str:
    return "ABCD"[int(row["answer_index"])]
