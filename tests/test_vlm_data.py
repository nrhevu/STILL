from pathlib import Path

from neural_kv.data.vlm import (
    format_vlm_mcq_prompt,
    normalize_mmmu_row,
    normalize_scienceqa_row,
    parse_answer_index,
    parse_options,
    vlm_answer_letter,
)
from neural_kv.training.vlm import extract_vlm_answer_letter
from neural_kv.training.vlm import build_vlm_messages


class FakeImage:
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fake-image")


def test_parse_options_accepts_stringified_mmmu_options() -> None:
    assert parse_options("['red', 'blue', 'green']") == ["red", "blue", "green"]


def test_parse_answer_index_accepts_letter_and_choice_text() -> None:
    choices = ["red", "blue", "green"]

    assert parse_answer_index("B", choices) == 1
    assert parse_answer_index("green", choices) == 2


def test_normalize_scienceqa_row_saves_image_without_solution_leak(tmp_path) -> None:
    row = {
        "image": FakeImage(),
        "question": "Which object is shown?",
        "choices": ["rock", "leaf", "star", "moon"],
        "answer": 1,
        "hint": "Use the picture.",
        "solution": "The picture shows a leaf.",
        "subject": "natural science",
    }

    normalized = normalize_scienceqa_row(
        row,
        split="validation",
        row_index=0,
        output_dir=tmp_path,
    )

    assert normalized is not None
    assert normalized["source"] == "scienceqa"
    assert normalized["answer_letter"] == "B"
    assert len(normalized["images"]) == 1
    assert normalized["images"][0].startswith("images/validation/")
    assert normalized["images"][0].endswith("_0.png")
    assert (tmp_path / normalized["images"][0]).exists()
    prompt = format_vlm_mcq_prompt(normalized)
    assert "Use the picture." in prompt
    assert "solution" not in prompt.lower()
    assert "leaf" in prompt


def test_normalize_mmmu_row_parses_letter_answer_and_images(tmp_path) -> None:
    row = {
        "id": "validation_Physics_1",
        "question": "What does the diagram show? <image 1>",
        "options": "['force', 'mass', 'charge']",
        "answer": "C",
        "image_1": FakeImage(),
        "question_type": "multiple-choice",
        "subfield": "Mechanics",
    }

    normalized = normalize_mmmu_row(
        row,
        split="validation",
        subject="Physics",
        row_index=0,
        output_dir=tmp_path,
    )

    assert normalized is not None
    assert normalized["question"] == "What does the diagram show?"
    assert normalized["answer_index"] == 2
    assert vlm_answer_letter(normalized) == "C"
    assert normalized["images"] == ["images/validation/validation_Physics_1_0.png"]


def test_official_mmmu_prompt_matches_report_shape() -> None:
    row = {
        "question": "What does the diagram show?",
        "choices": ["force", "mass", "charge"],
        "context_text": "",
    }

    prompt = format_vlm_mcq_prompt(row, prompt_style="official_mmmu")

    assert "Question: What does the diagram show?" in prompt
    assert "Options:\n\nA. force\nB. mass\nC. charge" in prompt
    assert prompt.endswith("Please select the correct answer from the options above.")
    assert "Answer:" not in prompt


def test_qwen_mmmu_prompt_matches_lmms_eval_shape() -> None:
    row = {
        "question": "What does the diagram show?",
        "choices": ["force", "mass", "charge"],
        "context_text": "",
    }

    prompt = format_vlm_mcq_prompt(row, prompt_style="qwen_mmmu")

    assert prompt == (
        "Question: What does the diagram show?\n"
        "Options:\n"
        "A. force\n"
        "B. mass\n"
        "C. charge\n"
        "Answer with the option letter only."
    )


def test_extract_vlm_answer_letter_handles_generation_text() -> None:
    row = {"choices": ["$6", "$7", "$8", "$9"]}

    assert extract_vlm_answer_letter("The correct answer is B.", row) == "B"
    assert extract_vlm_answer_letter("After checking, final answer: (C)", row) == "C"
    assert extract_vlm_answer_letter("I choose option D because the table says so.", row) == "D"


def test_build_vlm_messages_can_include_system_prompt(tmp_path) -> None:
    row = {
        "question": "What is shown?",
        "choices": ["cat", "dog"],
        "images": [],
    }

    messages = build_vlm_messages(
        row,
        base_dir=tmp_path,
        prompt_style="qwen_mmmu",
        system_prompt="You are a helpful assistant.",
    )

    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }
    assert messages[1]["role"] == "user"
