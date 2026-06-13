from types import SimpleNamespace

import torch

from neural_kv.hf_training import lexical_query_exact_token_indices
from neural_kv.niah import make_niah_case, niah_question, token_ids


class CharTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(int(item)) for item in ids)


def test_two_hop_case_separates_query_key_from_answer_record() -> None:
    tokenizer = CharTokenizer()
    case = make_niah_case(
        tokenizer,
        context_length=4096,
        depth_percent=25,
        trial=0,
        seed=123,
        task="two_hop",
    )

    route_fragment = f"retrieval key {case.key} maps to vault key {case.secondary_key}"
    vault_fragment = f"vault key {case.secondary_key} is {case.value}"

    assert case.task == "two_hop"
    assert route_fragment in case.context
    assert vault_fragment in case.context
    assert case.context.find(route_fragment) != case.context.find(vault_fragment)
    assert abs(len(token_ids(tokenizer, case.context)) - 4096) <= 2


def test_two_hop_lexical_exact_selector_does_not_trivially_include_answer() -> None:
    tokenizer = CharTokenizer()
    case = make_niah_case(
        tokenizer,
        context_length=4096,
        depth_percent=25,
        trial=1,
        seed=123,
        task="two_hop",
    )
    context_ids = token_ids(tokenizer, case.context)
    indices = lexical_query_exact_token_indices(
        tokenizer,
        {"context": case.context, "question": niah_question(case)},
        torch.tensor([context_ids]),
        max_tokens=96,
        device="cpu",
    )

    assert indices is not None
    selected = "".join(case.context[index] for index in indices.tolist())
    assert case.key in selected
    assert case.value not in selected


def test_two_hop_linked_lexical_selector_includes_answer_record() -> None:
    tokenizer = CharTokenizer()
    case = make_niah_case(
        tokenizer,
        context_length=4096,
        depth_percent=25,
        trial=2,
        seed=123,
        task="two_hop",
    )
    context_ids = token_ids(tokenizer, case.context)
    indices = lexical_query_exact_token_indices(
        tokenizer,
        {"context": case.context, "question": niah_question(case)},
        torch.tensor([context_ids]),
        max_tokens=512,
        device="cpu",
        include_linked=True,
    )

    assert indices is not None
    selected = "".join(case.context[index] for index in indices.tolist())
    assert case.key in selected
    assert case.secondary_key in selected
    assert case.value in selected


def test_single_lexical_selector_ignores_generic_secret_key_filler() -> None:
    tokenizer = CharTokenizer()
    case = make_niah_case(
        tokenizer,
        context_length=4096,
        depth_percent=0,
        trial=0,
        seed=123,
        task="single",
    )
    context_ids = token_ids(tokenizer, case.context)

    indices = lexical_query_exact_token_indices(
        tokenizer,
        {"context": case.context, "question": niah_question(case)},
        torch.tensor([context_ids]),
        max_tokens=512,
        device="cpu",
        include_linked=True,
    )

    assert indices is not None
    selected = "".join(case.context[index] for index in indices.tolist())
    assert case.key in selected
    assert case.value in selected
    assert "Archive line" not in selected


def test_multi_needle_lexical_selector_ignores_distractor_niah_records() -> None:
    tokenizer = CharTokenizer()
    case = make_niah_case(
        tokenizer,
        context_length=4096,
        depth_percent=100,
        trial=0,
        seed=123,
        task="multi_needle",
    )
    context_ids = token_ids(tokenizer, case.context)

    indices = lexical_query_exact_token_indices(
        tokenizer,
        {"context": case.context, "question": niah_question(case)},
        torch.tensor([context_ids]),
        max_tokens=512,
        device="cpu",
        include_linked=True,
    )

    assert indices is not None
    selected = "".join(case.context[index] for index in indices.tolist())
    assert case.key in selected
    assert case.value in selected
    assert "NIAH-DISTRACTOR" not in selected
