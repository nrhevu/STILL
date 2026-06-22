from types import SimpleNamespace

import torch

from neural_kv.data.ruler import build_ruler_mcq_examples
from neural_kv.hf_training import (
    _fresh_dynamic_cache,
    extract_answer_letter,
    kl_and_ce_loss,
    letter_kl_and_ce_loss,
    lexical_query_exact_token_indices,
    load_model_and_tokenizer,
)
from neural_kv.training import distillation


class CharTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def test_load_model_and_tokenizer_device_map_auto_skips_global_to(monkeypatch) -> None:
    calls = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeModel:
        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, device):
            calls["to"] = device
            return self

        def eval(self) -> None:
            calls["eval"] = True

        def parameters(self):
            return [self.weight]

    def fake_tokenizer_from_pretrained(model_name, *, use_fast):
        calls["tokenizer"] = (model_name, use_fast)
        return FakeTokenizer()

    def fake_model_from_pretrained(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return FakeModel()

    monkeypatch.setattr(
        distillation.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        distillation.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    model, tokenizer = load_model_and_tokenizer(
        "huge-model",
        device="cuda",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    assert tokenizer.pad_token == "<eos>"
    assert calls["model"][0] == "huge-model"
    assert calls["model"][1]["device_map"] == "auto"
    assert calls["model"][1]["torch_dtype"] is torch.bfloat16
    assert "to" not in calls
    assert calls["eval"] is True
    assert model.weight.requires_grad is False


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


def test_fresh_dynamic_cache_update_does_not_mutate_original_cache() -> None:
    from transformers.cache_utils import DynamicCache

    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    original = DynamicCache.from_legacy_cache(((key, value),))
    fresh = _fresh_dynamic_cache(original)

    fresh.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), 0)

    assert original.get_seq_length() == 3
    assert fresh.get_seq_length() == 4


def test_extract_answer_letter_prefers_explicit_tail_answer() -> None:
    text = "<think>I considered option A first.</think>\n\nAnswer: C"

    assert extract_answer_letter(text) == "C"


def test_extract_answer_letter_accepts_final_standalone_letter() -> None:
    assert extract_answer_letter("After checking the context, the answer is\nD\n") == "D"


def test_lexical_query_exact_token_indices_selects_matching_context_line() -> None:
    context = "\n".join(
        [
            "Company: Microsoft Corp. | Concept: Revenue | Fiscal year: 2024 | "
            "Period: FY | Unit: USD | Value: 456",
            "Company: Apple Inc. | Concept: Revenue | Fiscal year: 2024 | "
            "Period: FY | Unit: USD | Value: 123",
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


def test_lexical_query_exact_token_indices_selects_ruler_target_line() -> None:
    row = build_ruler_mcq_examples(
        split="validation",
        count=1,
        context_tokens=256,
        tasks=("niah_single",),
        seed=13,
        target_placement="middle",
    )[0]
    context = str(row["context"])
    context_ids = torch.tensor([[ord(char) for char in context]])

    indices = lexical_query_exact_token_indices(
        CharTokenizer(),
        row,
        context_ids,
        max_tokens=1024,
        device="cpu",
    )

    assert indices is not None
    selected = "".join(chr(int(context_ids[0, index])) for index in indices.tolist())
    assert str(row["target_line"]) in selected
    assert str(row["answer"]) in selected
    assert f"Correct option label {row['answer_letter']}" in selected
