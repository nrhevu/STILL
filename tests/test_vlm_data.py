from PIL import Image

from neural_kv.data.vlm import (
    VLMExample,
    extract_choice_letter,
    format_vlm_source_prompt,
    relaxed_answer_match,
    resize_image_max_side,
    score_vlm_prediction,
    vlm_example_from_hf_row,
)


def test_vlm_example_from_mmmu_style_row_scores_letter() -> None:
    image = Image.new("RGB", (100, 50), "white")
    row = {
        "id": "sample-1",
        "question": "Which curve is highest?",
        "options": ["Alpha", "Beta", "Gamma", "Delta"],
        "answer": "C",
        "image_1": image,
    }

    example = vlm_example_from_hf_row(
        row,
        dataset="MMMU/MMMU",
        task="mmmu",
        split="validation",
        subject="Accounting",
    )

    prompt = format_vlm_source_prompt(example)
    prediction, correct = score_vlm_prediction(example, "Answer: C")
    assert example.answer_letter == "C"
    assert "Options:" in prompt
    assert prediction == "C"
    assert correct is True


def test_extract_choice_letter_prefers_explicit_answer_tail() -> None:
    assert extract_choice_letter("I first thought B.\nFinal answer: D") == "D"


def test_relaxed_answer_match_accepts_numeric_tolerance() -> None:
    assert relaxed_answer_match("104.9", ["100"], relative_tolerance=0.05)
    assert not relaxed_answer_match("111", ["100"], relative_tolerance=0.05)


def test_resize_image_max_side_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (200, 100), "white")
    resized = resize_image_max_side(image, 50)
    assert resized.size == (50, 25)


def test_docvqa_style_row_uses_free_form_answers() -> None:
    image = Image.new("RGB", (32, 32), "white")
    example = vlm_example_from_hf_row(
        {
            "questionId": 7,
            "question": "What is the invoice total?",
            "answers": ["$42.00", "42"],
            "image": image,
        },
        dataset="lmms-lab/DocVQA",
        task="docvqa",
        split="validation",
    )

    prediction, correct = score_vlm_prediction(example, "Answer: 42")
    assert isinstance(example, VLMExample)
    assert example.choices is None
    assert prediction == "42"
    assert correct is True
