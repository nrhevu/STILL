"""Vision-language benchmark adapters for neural KV evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from neural_kv.data.mcq import stable_id

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9.+\-/% ]+")
_ANSWER_PREFIX_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:\-]\s*")
_LETTER_RE = re.compile(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?%?")


@dataclass(frozen=True)
class VLMExample:
    """One normalized VLM benchmark row."""

    id: str
    dataset: str
    task: str
    question: str
    image: Any
    answers: list[str]
    choices: list[str] | None = None
    answer_index: int | None = None
    split: str = ""
    subject: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)

    @property
    def answer_letter(self) -> str | None:
        if self.answer_index is None:
            return None
        if not 0 <= self.answer_index < 26:
            return None
        return chr(ord("A") + self.answer_index)


def normalize_vlm_answer(text: object) -> str:
    """Normalize a free-form VQA answer for exact matching."""
    normalized = str(text).strip().lower()
    normalized = _ANSWER_PREFIX_RE.split(normalized)[-1]
    normalized = normalized.splitlines()[0] if normalized else ""
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def extract_choice_letter(text: object) -> str | None:
    """Extract an A-D answer letter from generated text."""
    value = str(text).strip().upper()
    matches = list(_ANSWER_PREFIX_RE.finditer(value))
    if matches:
        value = value[matches[-1].end() :]
    tail = value[-256:]
    letter_matches = list(_LETTER_RE.finditer(tail))
    if not letter_matches:
        return None
    return letter_matches[-1].group(1).upper()


def extract_short_answer(text: object) -> str:
    """Extract a concise generated answer span."""
    value = str(text).strip()
    parts = _ANSWER_PREFIX_RE.split(value)
    if len(parts) > 1:
        value = parts[-1]
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return value.strip()


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0).replace(",", "")
        if token in {"", "+", "-", ".", "%"}:
            continue
        is_percent = token.endswith("%")
        token = token.rstrip("%")
        try:
            value = float(token)
        except ValueError:
            continue
        values.append(value / 100.0 if is_percent else value)
    return values


def relaxed_answer_match(
    prediction: object,
    answers: Iterable[object],
    *,
    relative_tolerance: float = 0.05,
) -> bool:
    """Return true for exact normalized matches or ChartQA-style numeric tolerance."""
    pred_norm = normalize_vlm_answer(prediction)
    if not pred_norm:
        return False
    pred_numbers = _numbers(pred_norm)
    for answer in answers:
        answer_norm = normalize_vlm_answer(answer)
        if pred_norm == answer_norm:
            return True
        answer_numbers = _numbers(answer_norm)
        if pred_numbers and answer_numbers:
            for pred_value in pred_numbers:
                for answer_value in answer_numbers:
                    tolerance = max(abs(answer_value) * relative_tolerance, 1e-6)
                    if math.isclose(pred_value, answer_value, rel_tol=0.0, abs_tol=tolerance):
                        return True
    return False


def score_vlm_prediction(example: VLMExample, generated_text: object) -> tuple[str | None, bool]:
    """Normalize and score generated text for an example."""
    if example.is_multiple_choice:
        prediction = extract_choice_letter(generated_text)
        return prediction, prediction is not None and prediction == example.answer_letter
    prediction = extract_short_answer(generated_text)
    return prediction, relaxed_answer_match(prediction, example.answers)


def format_vlm_source_prompt(example: VLMExample, *, no_think: bool = True) -> str:
    """Build the image-grounded source prompt, excluding the final answer cue."""
    thinking = "/no_think\n" if no_think else ""
    if example.is_multiple_choice:
        assert example.choices is not None
        labels = [chr(ord("A") + idx) for idx in range(len(example.choices))]
        rendered = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, example.choices, strict=True)
        )
        return (
            thinking
            + "Use the image to answer the question.\n\n"
            + f"Question: {example.question}\n\n"
            + f"Options:\n{rendered}\n\n"
            + "Answer with only the single capital letter of the correct option."
        )
    return (
        thinking
        + "Use the image to answer the question. Provide a short answer.\n\n"
        + f"Question: {example.question}"
    )


def vlm_answer_prompt(example: VLMExample) -> str:
    """Text-only continuation prompt fed after the cached multimodal prefix."""
    del example
    return "\nAnswer:"


def resize_image_max_side(image: Any, max_side: int | None) -> Any:
    """Resize a PIL-like image while preserving aspect ratio."""
    if max_side is None or max_side <= 0:
        return image
    size = getattr(image, "size", None)
    if not size or len(size) != 2:
        return image
    width, height = int(size[0]), int(size[1])
    current = max(width, height)
    if current <= 0 or current == max_side:
        return image
    scale = max_side / current
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resampling = getattr(getattr(image, "Resampling", object), "BICUBIC", None)
    if resampling is None:
        try:
            from PIL import Image

            resampling = Image.Resampling.BICUBIC
        except Exception:
            resampling = 3
    return image.resize(new_size, resampling)


def _first_present(row: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def _coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                pass
        return [stripped]
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_choices(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                pass
        pieces = [piece.strip() for piece in re.split(r"\n|;\s*", stripped) if piece.strip()]
        return pieces or None
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [str(item) for item in value]
    return None


def _answer_index_from_value(answer: Any, choices: list[str] | None) -> int | None:
    if answer is None:
        return None
    if isinstance(answer, int):
        return answer
    text = str(answer).strip()
    if len(text) == 1 and "A" <= text.upper() <= "Z":
        return ord(text.upper()) - ord("A")
    if text.isdigit():
        return int(text)
    if choices is not None:
        lowered = text.lower()
        for idx, choice in enumerate(choices):
            if choice.lower() == lowered:
                return idx
    return None


def _first_image(row: dict[str, Any]) -> Any:
    value = _first_present(row, ("image", "img", "picture"))
    if value is not None:
        return value
    for key in sorted(row):
        if key.startswith("image_") and row[key] is not None:
            return row[key]
    images = row.get("images")
    if isinstance(images, Sequence) and images:
        return images[0]
    return None


def vlm_example_from_hf_row(
    row: dict[str, Any],
    *,
    dataset: str,
    task: str,
    split: str,
    subject: str = "",
) -> VLMExample:
    """Normalize common MMMU, DocVQA, and ChartQA row shapes."""
    question = _first_present(row, ("question", "query", "Question", "question_text"))
    if question is None:
        raise ValueError(f"Could not find a question field in {dataset} row")
    image = _first_image(row)
    if image is None:
        raise ValueError(f"Could not find an image field in {dataset} row")

    choices = _coerce_choices(_first_present(row, ("choices", "options", "all_choices")))
    raw_answer = _first_present(row, ("answer", "answers", "label", "gt_answer"))
    answer_index = _answer_index_from_value(raw_answer, choices)
    if choices is not None and answer_index is not None and 0 <= answer_index < len(choices):
        answers = [choices[answer_index]]
    else:
        answers = _coerce_answers(raw_answer)

    row_id = _first_present(row, ("id", "question_id", "questionId", "qid"))
    if row_id is None:
        row_id = stable_id(dataset, split, subject, str(question), str(raw_answer))
    return VLMExample(
        id=str(row_id),
        dataset=dataset,
        task=task,
        question=str(question),
        image=image,
        answers=answers,
        choices=choices,
        answer_index=answer_index,
        split=split,
        subject=subject,
        metadata={
            key: value
            for key, value in row.items()
            if key
            not in {
                "image",
                "img",
                "picture",
                "images",
                "question",
                "query",
                "Question",
                "choices",
                "options",
                "answer",
                "answers",
                "label",
            }
            and not key.startswith("image_")
        },
    )


def load_hf_vlm_examples(
    *,
    dataset_name: str,
    task: str,
    split: str,
    dataset_config: str | None = None,
    limit: int | None = None,
) -> list[VLMExample]:
    """Load normalized VLM examples from Hugging Face Datasets."""
    from neural_kv.utils.hf_cache import configure_hf_cache

    configure_hf_cache()

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    examples: list[VLMExample] = []
    for row in dataset:
        examples.append(
            vlm_example_from_hf_row(
                dict(row),
                dataset=dataset_name,
                task=task,
                split=split,
                subject=dataset_config or "",
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def iter_hf_vlm_examples_from_specs(
    specs: Iterable[dict[str, Any]],
    *,
    limit_per_dataset: int | None = None,
) -> Iterator[VLMExample]:
    """Yield examples for a list of benchmark dataset config mappings."""
    from neural_kv.utils.hf_cache import configure_hf_cache

    configure_hf_cache()

    from datasets import get_dataset_config_names

    for spec in specs:
        task = str(spec.get("task") or spec.get("name") or "vlm")
        dataset_name = str(spec.get("hf_name") or spec.get("dataset_name") or spec["name"])
        split = str(spec.get("split", "validation"))
        configs = spec.get("configs", spec.get("config", None))
        if isinstance(configs, str):
            configs = [configs]
        if configs is None:
            configs = [None]
        if list(configs) == ["all"]:
            configs = get_dataset_config_names(dataset_name)
        remaining = limit_per_dataset
        for config in configs:
            examples = load_hf_vlm_examples(
                dataset_name=dataset_name,
                task=task,
                split=split,
                dataset_config=config,
                limit=remaining,
            )
            yield from examples
            if remaining is not None:
                remaining -= len(examples)
                if remaining <= 0:
                    break
