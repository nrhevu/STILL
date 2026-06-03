from neural_kv.data import build_mcq_examples, format_mcq_prompt


def test_build_mcq_examples_from_text() -> None:
    text = (
        "Ada Lovelace wrote notes about the Analytical Engine in 1843. "
        "The document also mentions Charles Babbage and mathematics. "
    ) * 10
    rows = build_mcq_examples(
        texts=[text],
        split="train",
        source="unit",
        max_docs=1,
        questions_per_doc=2,
        context_chars=2000,
        seed=1,
    )
    assert rows
    prompt = format_mcq_prompt(rows[0].__dict__)
    assert "Question:" in prompt
    assert "A." in prompt
