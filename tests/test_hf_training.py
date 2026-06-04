import torch
from types import SimpleNamespace

from neural_kv.hf_training import (
    extract_answer_letter,
    kl_and_ce_loss,
    letter_kl_and_ce_loss,
    lexical_query_exact_token_indices,
)


class CharTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def test_letter_kl_and_ce_loss_rewards_gold_letter() -> None:
    teacher_logits = torch.tensor([[0.0, 3.0, -1.0, -2.0]])
    good_student = torch.tensor([[0.0, 2.0, -1.0, -2.0]])
    bad_student = torch.tensor([[3.0, 0.0, -1.0, -2.0]])

    good_loss = letter_kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=good_student,
        target_index=1,
        kl_weight=1.0,
        ce_weight=1.0,
    )
    bad_loss = letter_kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=bad_student,
        target_index=1,
        kl_weight=1.0,
        ce_weight=1.0,
    )

    assert good_loss < bad_loss


def test_reverse_kl_loss_rewards_teacher_matching_distribution() -> None:
    teacher_logits = torch.tensor([[0.0, 3.0, -1.0, -2.0]])
    good_student = torch.tensor([[0.0, 2.5, -1.0, -2.0]])
    bad_student = torch.tensor([[3.0, 0.0, -1.0, -2.0]])

    good_loss = kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=good_student,
        target_ids=torch.tensor([[1]]),
        kl_weight=0.0,
        ce_weight=0.0,
        reverse_kl_weight=1.0,
    )
    bad_loss = kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=bad_student,
        target_ids=torch.tensor([[1]]),
        kl_weight=0.0,
        ce_weight=0.0,
        reverse_kl_weight=1.0,
    )

    assert good_loss < bad_loss


def test_extract_answer_letter_prefers_explicit_tail_answer() -> None:
    text = "<think>I considered option A first.</think>\n\nAnswer: C"

    assert extract_answer_letter(text) == "C"


def test_extract_answer_letter_accepts_final_standalone_letter() -> None:
    assert extract_answer_letter("After checking the context, the answer is\nD\n") == "D"


def test_lexical_query_exact_token_indices_selects_matching_context_line() -> None:
    context = "\n".join(
        [
            "Company: Microsoft Corp. | Concept: Revenue | Fiscal year: 2024 | Period: FY | Unit: USD | Value: 456",
            "Company: Apple Inc. | Concept: Revenue | Fiscal year: 2024 | Period: FY | Unit: USD | Value: 123",
        ]
    )
    row = {
        "context": context,
        "question": (
            "For Apple Inc., concept Revenue, fiscal year 2024, period FY, "
            "unit USD, what is the reported value?"
        ),
    }
    context_ids = torch.tensor([[ord(char) for char in context]])

    indices = lexical_query_exact_token_indices(
        CharTokenizer(),
        row,
        context_ids,
        max_tokens=128,
        device="cpu",
    )

    assert indices is not None
    selected = "".join(chr(int(context_ids[0, index])) for index in indices.tolist())
    assert "Apple Inc." in selected
    assert "Value: 123" in selected
