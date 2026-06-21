import random
from pathlib import Path

from neural_kv.training.datamodule import (
    BalancedAnswerSampler,
    JsonlMCQDataset,
    LegacyRandomSampler,
)


def _write_rows(path: Path, answers: list[int]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        for index, answer in enumerate(answers):
            row = {
                "id": str(index),
                "split": "train",
                "source": "unit",
                "context": "context",
                "question": "question",
                "choices": ["A", "B", "C", "D"],
                "answer_index": answer,
                "answer": "ABCD"[answer],
            }
            handle.write(json.dumps(row) + "\n")


def test_legacy_random_sampler_matches_train_still_schedule(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_rows(path, [0, 1, 2, 3, 0])
    dataset = JsonlMCQDataset(path)

    sampler = LegacyRandomSampler(dataset, num_samples=8, seed=7)

    rng = random.Random(7)
    expected = [rng.randrange(len(dataset)) for _ in range(8)]
    assert list(sampler) == expected


def test_balanced_sampler_matches_train_still_flattened_schedule(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_rows(path, [0, 1, 1, 2, 3, 3])
    dataset = JsonlMCQDataset(path)

    sampler = BalancedAnswerSampler(dataset, num_samples=8, seed=11)

    answer_groups = {
        "A": [0],
        "B": [1, 2],
        "C": [3],
        "D": [4, 5],
    }
    rng = random.Random(11)
    letters = sorted(answer_groups)
    expected = [rng.choice(answer_groups[rng.choice(letters)]) for _ in range(8)]
    assert list(sampler) == expected
