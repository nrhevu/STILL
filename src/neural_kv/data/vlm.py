"""VLM multiple-choice dataset preparation helpers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from neural_kv.data.mcq import stable_id

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_FIELDS = tuple(f"image_{index}" for index in range(1, 8))


def choice_labels(count: int) -> list[str]:
    if count <= 0 or count > len(CHOICE_LABELS):
        raise ValueError(f"choice count must be in [1, {len(CHOICE_LABELS)}]")
    return list(CHOICE_LABELS[:count])


def vlm_answer_letter(row: dict[str, Any]) -> str:
    return CHOICE_LABELS[int(row["answer_index"])]


def parse_options(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    raise ValueError("options/choices must be a list or a stringified list")


def parse_answer_index(value: object, choices: list[str]) -> int:
    if isinstance(value, int):
        index = value
    elif isinstance(value, str):
        stripped = value.strip()
        if len(stripped) == 1 and stripped.upper() in CHOICE_LABELS:
            index = CHOICE_LABELS.index(stripped.upper())
        elif stripped.isdigit():
            index = int(stripped)
        else:
            try:
                index = choices.index(stripped)
            except ValueError as exc:
                raise ValueError(f"answer {value!r} does not match any choice") from exc
    else:
        raise ValueError(f"Unsupported answer value: {value!r}")
    if index < 0 or index >= len(choices):
        raise ValueError(f"answer index {index} out of range for {len(choices)} choices")
    return index


def _is_missing_image(value: object) -> bool:
    return value is None or value == "" or value == "Not supported with pagination yet"


def has_image(row: dict[str, Any]) -> bool:
    if not _is_missing_image(row.get("image")):
        return True
    return any(not _is_missing_image(row.get(field)) for field in IMAGE_FIELDS)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "row"


def _save_image(value: object, path: Path) -> bool:
    if _is_missing_image(value):
        return False
    if not hasattr(value, "save"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image = value
    if getattr(image, "mode", None) not in {None, "RGB"} and hasattr(image, "convert"):
        image = image.convert("RGB")
    image.save(path)
    return True


def save_row_images(
    row: dict[str, Any],
    *,
    output_dir: str | Path,
    split: str,
    row_id: str,
) -> list[str]:
    output_path = Path(output_dir)
    image_dir = output_path / "images" / split
    saved: list[str] = []
    candidates = [("image", row.get("image"))]
    candidates.extend((field, row.get(field)) for field in IMAGE_FIELDS)
    for _, value in candidates:
        image_index = len(saved)
        filename = f"{_safe_filename(row_id)}_{image_index}.png"
        path = image_dir / filename
        if _save_image(value, path):
            saved.append(str(path.relative_to(output_path)))
    return saved


def normalize_scienceqa_row(
    row: dict[str, Any],
    *,
    split: str,
    row_index: int,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    if not has_image(row):
        return None
    choices = parse_options(row.get("choices"))
    answer_index = parse_answer_index(row.get("answer"), choices)
    row_id = str(row.get("id") or stable_id("scienceqa", split, str(row_index)))
    images = (
        save_row_images(row, output_dir=output_dir, split=split, row_id=row_id)
        if output_dir is not None
        else []
    )
    if output_dir is not None and not images:
        return None
    context_text = str(row.get("hint") or "").strip()
    example = {
        "id": row_id,
        "source": "scienceqa",
        "split": split,
        "images": images,
        "context_text": context_text,
        "question": str(row["question"]),
        "choices": choices,
        "answer_index": answer_index,
        "answer_letter": CHOICE_LABELS[answer_index],
        "task": str(row.get("task") or ""),
        "subject": str(row.get("subject") or ""),
        "topic": str(row.get("topic") or ""),
        "category": str(row.get("category") or ""),
        "skill": str(row.get("skill") or ""),
    }
    return example


def normalize_mmmu_row(
    row: dict[str, Any],
    *,
    split: str,
    subject: str,
    row_index: int,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    if not has_image(row):
        return None
    try:
        choices = parse_options(row.get("options"))
        answer_index = parse_answer_index(row.get("answer"), choices)
    except ValueError:
        return None
    row_id = str(row.get("id") or stable_id("mmmu", subject, split, str(row_index)))
    images = (
        save_row_images(row, output_dir=output_dir, split=split, row_id=row_id)
        if output_dir is not None
        else []
    )
    if output_dir is not None and not images:
        return None
    question = str(row["question"]).replace("<image 1>", "").strip()
    for index in range(2, 8):
        question = question.replace(f"<image {index}>", "").strip()
    example = {
        "id": row_id,
        "source": "mmmu",
        "split": split,
        "images": images,
        "context_text": "",
        "question": question,
        "choices": choices,
        "answer_index": answer_index,
        "answer_letter": CHOICE_LABELS[answer_index],
        "task": str(row.get("question_type") or ""),
        "subject": subject,
        "topic": str(row.get("subfield") or ""),
        "category": str(row.get("img_type") or ""),
        "difficulty": str(row.get("topic_difficulty") or ""),
    }
    return example


def format_vlm_mcq_prompt(row: dict[str, Any], *, prompt_style: str = "compact") -> str:
    choices = parse_options(row["choices"])
    labels = choice_labels(len(choices))
    rendered_choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
    )
    context = str(row.get("context_text") or "").strip()
    context_block = f"Context: {context}\n\n" if context else ""
    if prompt_style == "official_mmmu":
        if context_block:
            question = f"{context_block}Question: {row['question']}"
        else:
            question = f"Question: {row['question']}"
        return (
            f"{question}\n\n"
            f"Options:\n\n{rendered_choices}\n\n"
            "Please select the correct answer from the options above."
        )
    if prompt_style == "qwen_mmmu":
        if context_block:
            question = f"{context_block}Question: {row['question']}"
        else:
            question = f"Question: {row['question']}"
        return (
            f"{question}\n"
            f"Options:\n{rendered_choices}\n"
            "Answer with the option letter only."
        )
    if prompt_style != "compact":
        raise ValueError(f"Unsupported VLM prompt_style: {prompt_style}")
    return (
        f"{context_block}"
        f"Question: {row['question']}\n\n"
        f"Options:\n{rendered_choices}\n\n"
        "Answer with only the single capital letter of the correct option.\n"
        "Answer:"
    )


def resolve_image_paths(row: dict[str, Any], *, base_dir: str | Path) -> list[str]:
    base_path = Path(base_dir)
    paths: list[str] = []
    for image in row.get("images", []):
        image_path = Path(str(image))
        if not image_path.is_absolute():
            image_path = base_path / image_path
        paths.append(str(image_path))
    return paths


def write_vlm_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return len(rows)
