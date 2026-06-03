from neural_kv.data import build_mcq_examples, chunk_texts, format_mcq_prompt


def test_build_mcq_examples_from_text() -> None:
    text = (
        "Ada Lovelace wrote notes about the Analytical Engine in 1843. "
        "The document also mentions Charles Babbage and mathematics. "
    ) * 10
    distractor_text = (
        "Marie Curie studied radium in Paris with Pierre Curie. "
        "The laboratory record mentions chemistry and Nobel prizes. "
    ) * 10
    rows = build_mcq_examples(
        texts=[text, distractor_text],
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


def test_chunk_texts_expands_long_sources() -> None:
    text = "".join(str(idx % 10) for idx in range(1000))
    chunks = chunk_texts([text], context_chars=200, chunks_per_text=3, stride_chars=100)
    assert len(chunks) == 3
    assert chunks[0] == text[:200]
    assert chunks[1] == text[100:300]
