"""Lightning data modules for JSONL MCQ rows."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from neural_kv.data import answer_letter, read_jsonl

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - exercised only without train extra
    pl = None  # type: ignore[assignment]


class JsonlMCQDataset(Dataset[dict[str, Any]]):
    """Map-style dataset backed by MCQ JSONL rows."""

    def __init__(self, path: str | Path, *, limit: int | None = None) -> None:
        self.path = Path(path)
        self.rows = read_jsonl(self.path, limit=limit)
        if not self.rows:
            raise ValueError(f"No rows found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class LegacyRandomSampler(torch.utils.data.Sampler[int]):
    """Sample row indices with the exact Python RNG schedule used by legacy training."""

    def __init__(self, dataset: JsonlMCQDataset, *, num_samples: int, seed: int) -> None:
        self.dataset_length = len(dataset)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        if self.dataset_length <= 0:
            raise ValueError("Legacy random sampling requires a non-empty dataset")

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_samples):
            yield rng.randrange(self.dataset_length)

    def __len__(self) -> int:
        return self.num_samples


class BalancedAnswerSampler(torch.utils.data.Sampler[int]):
    """Sample rows with replacement after first sampling an answer letter."""

    def __init__(self, dataset: JsonlMCQDataset, *, num_samples: int, seed: int) -> None:
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            groups[answer_letter(row)].append(index)
        self.groups = {letter: indices for letter, indices in groups.items() if indices}
        if not self.groups:
            raise ValueError("Balanced sampling requires at least one answer group")
        self.letters = sorted(self.groups)

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_samples):
            letter = rng.choice(self.letters)
            yield rng.choice(self.groups[letter])

    def __len__(self) -> int:
        return self.num_samples


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


class NeuralKVDataModule(pl.LightningDataModule if pl is not None else object):
    """JSONL data module with legacy-compatible replacement sampling."""

    def __init__(
        self,
        *,
        train_file: str,
        validation_file: str | None = None,
        batch_size: int = 1,
        steps_per_epoch: int | None = None,
        limit_train: int | None = None,
        limit_validation: int | None = None,
        seed: int = 7,
        balanced_answer_sampling: bool = False,
        num_workers: int = 0,
    ) -> None:
        if pl is None:
            raise ModuleNotFoundError("Install the train extra to use Lightning data modules")
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.batch_size = int(batch_size)
        self.steps_per_epoch = steps_per_epoch
        self.limit_train = limit_train
        self.limit_validation = limit_validation
        self.seed = int(seed)
        self.balanced_answer_sampling = bool(balanced_answer_sampling)
        self.num_workers = int(num_workers)
        self.train_dataset: JsonlMCQDataset | None = None
        self.validation_dataset: JsonlMCQDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = JsonlMCQDataset(self.train_file, limit=self.limit_train)
            if self.validation_file:
                self.validation_dataset = JsonlMCQDataset(
                    self.validation_file,
                    limit=self.limit_validation,
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must run before train_dataloader")
        num_samples = None
        if self.steps_per_epoch is not None:
            num_samples = int(self.steps_per_epoch) * self.batch_size
        if self.balanced_answer_sampling:
            if num_samples is None:
                num_samples = len(self.train_dataset)
            sampler = BalancedAnswerSampler(
                self.train_dataset,
                num_samples=num_samples,
                seed=self.seed,
            )
        elif num_samples is not None:
            sampler = LegacyRandomSampler(
                self.train_dataset,
                num_samples=num_samples,
                seed=self.seed,
            )
        else:
            sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_rows,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.validation_dataset is None:
            return None
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_rows,
            num_workers=self.num_workers,
        )
