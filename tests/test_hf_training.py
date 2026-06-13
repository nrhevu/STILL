from types import SimpleNamespace

import torch

from neural_kv.compactor import StillCompactor
from neural_kv.hf_training import (
    _fresh_dynamic_cache,
    extract_answer_letter,
    kl_and_ce_loss,
    letter_kl_and_ce_loss,
    lexical_query_exact_token_indices,
    parse_max_memory,
    place_compactor_for_model,
    prefill_context_cache,
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


def test_parse_max_memory_accepts_csv_and_json() -> None:
    assert parse_max_memory("0=280GiB,1=256GiB,cpu=512GiB") == {
        0: "280GiB",
        1: "256GiB",
        "cpu": "512GiB",
    }
    assert parse_max_memory('{"0":"280GiB","cpu":"512GiB"}') == {
        0: "280GiB",
        "cpu": "512GiB",
    }


def test_place_compactor_for_model_uses_decoder_layer_devices() -> None:
    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()

    compactor = StillCompactor(
        num_hidden_layers=2,
        head_dim=2,
        num_latents=1,
        rope_theta=10000.0,
    )

    placement = place_compactor_for_model(compactor, FakeModel(), fallback_device="cpu")

    assert placement == {0: "cpu", 1: "cpu"}
    assert str(next(compactor.layers[0].parameters()).device) == "cpu"


def test_prefill_context_cache_chunks_long_inputs() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, **kwargs):
            input_ids = kwargs["input_ids"]
            previous = int(kwargs.get("past_key_values") or 0)
            total = previous + int(input_ids.shape[-1])
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            self.calls.append(
                {
                    "tokens": int(input_ids.shape[-1]),
                    "previous": previous,
                    "attention_tokens": int(attention_mask.shape[-1])
                    if attention_mask is not None
                    else None,
                    "positions": position_ids.reshape(-1).tolist()
                    if position_ids is not None
                    else None,
                }
            )
            return SimpleNamespace(past_key_values=total)

    model = FakeModel()
    outputs = prefill_context_cache(model, torch.arange(5).reshape(1, 5), chunk_size=2)

    assert outputs.past_key_values == 5
    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert model.calls[0]["attention_tokens"] is None
    assert model.calls[1]["attention_tokens"] == 4
    assert model.calls[1]["positions"] == [2, 3]
    assert model.calls[2]["attention_tokens"] == 5
    assert model.calls[2]["positions"] == [4]


def test_compactor_zero_latents_keeps_explicit_exact_tokens_only() -> None:
    keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    values = keys + 100
    compactor = StillCompactor(
        num_hidden_layers=1,
        head_dim=2,
        num_latents=0,
        rope_theta=10000.0,
        exact_tokens=2,
        exact_strategy="lexical_linked",
        beta_base="zero",
    )

    compact = compactor(
        ((keys, values),),
        metadata={"source_tokens": 5},
        exact_token_indices=torch.tensor([1, 3]),
    )

    assert compact.num_tokens == 2
    assert torch.equal(compact.keys[0], keys[..., [1, 3], :])
    assert torch.equal(compact.values[0], values[..., [1, 3], :])
    assert torch.equal(compact.biases[0], torch.zeros(1, 1, 2))
    assert compact.metadata["latent_tokens"] == 0
