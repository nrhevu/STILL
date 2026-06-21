"""Small metric helpers used by training and evaluation."""

from __future__ import annotations


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def accuracy(correct: int, total: int) -> float:
    return float(correct) / float(total) if total else 0.0
