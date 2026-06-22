from neural_kv.data import DEFAULT_RULER_TASKS, build_ruler_mcq_examples, format_mcq_prompt


def test_build_ruler_mcq_examples_cover_tasks_and_schema() -> None:
    rows = build_ruler_mcq_examples(
        split="validation",
        count=len(DEFAULT_RULER_TASKS),
        context_tokens=192,
        tasks=DEFAULT_RULER_TASKS,
        seed=5,
        target_placement="middle",
        source="ruler_200k",
    )

    assert len(rows) == len(DEFAULT_RULER_TASKS)
    assert {row["ruler_task"] for row in rows} == set(DEFAULT_RULER_TASKS)
    for row in rows:
        answer = str(row["answer"])
        assert row["source"] == "ruler_200k"
        assert row["split"] == "validation"
        assert len(str(row["context"]).split()) == 192
        context = str(row["context"])
        target_line = str(row["target_line"])
        assert answer in context
        assert answer in target_line
        assert str(row["answer_letter"]) in "ABCD"
        assert f"Correct option label {row['answer_letter']}" in target_line
        assert target_line in context.splitlines()
        assert int(row["answer_char_offset"]) >= 0
        assert row["choices"][int(row["answer_index"])] == answer
        present_choices = [choice for choice in row["choices"] if str(choice) in context]
        assert present_choices == [answer]
        assert "Question:" in format_mcq_prompt(row)


def test_ruler_random_visible_keeps_target_before_visible_budget() -> None:
    rows = build_ruler_mcq_examples(
        split="test",
        count=8,
        context_tokens=192,
        tasks=("niah_single",),
        seed=9,
        target_placement="random_visible",
        visible_target_tokens=12,
    )

    assert rows
    assert all(int(row["target_word_offset"]) <= 12 for row in rows)


def test_ruler_tail_visible_keeps_target_near_context_end() -> None:
    rows = build_ruler_mcq_examples(
        split="test",
        count=8,
        context_tokens=192,
        tasks=("niah_single",),
        seed=11,
        target_placement="tail_visible",
        visible_target_tokens=12,
    )

    assert rows
    for row in rows:
        target_words = len(str(row["target_line"]).split())
        max_offset = 192 - target_words
        assert int(row["target_word_offset"]) >= max_offset - 20
        assert int(row["target_word_offset"]) <= max_offset - 8


def test_ruler_answer_letters_are_not_degenerate() -> None:
    rows = build_ruler_mcq_examples(
        split="train",
        count=64,
        context_tokens=192,
        seed=19,
    )

    assert {row["answer_letter"] for row in rows} == set("ABCD")


def test_ruler_choices_are_opaque_and_do_not_expose_semantic_answer() -> None:
    rows = build_ruler_mcq_examples(
        split="test",
        count=32,
        context_tokens=192,
        seed=29,
    )

    mismatched_suffix_rows = 0
    for row in rows:
        answer_index = int(row["answer_index"])
        answer = str(row["answer"])
        assert "_CHOICE_" in answer
        assert row["choices"][answer_index] == answer
        for choice in row["choices"]:
            text_choice = str(choice)
            assert "_CHOICE_" in text_choice
            assert "_DISTRACTOR_" not in text_choice
            assert "_VALUE_" not in text_choice
            assert "_ANSWER_" not in text_choice
            assert "_MARKER_" not in text_choice
            assert "_STATE_" not in text_choice
        suffix = int(answer.rsplit("_", 1)[1])
        mismatched_suffix_rows += int(suffix != answer_index)

    assert mismatched_suffix_rows > 0
