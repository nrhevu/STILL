import argparse
from pathlib import Path

from scripts.run_ruler_200k_acceptance import build_commands


def test_ruler_acceptance_runner_builds_strict_gate_commands() -> None:
    args = argparse.Namespace(
        checkpoint="checkpoints/run/step_100.pt",
        base_model="/models/qwen3-235b",
        raw_file="data/ruler_200k/test.jsonl",
        eval_file="data/ruler_200k/test.jsonl",
        output_dir="outputs/ruler_acceptance",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        context_length=200000,
        exact_tokens=1024,
        limit=64,
        score_mode="letter",
        device="auto",
        dtype="bfloat16",
        device_map_auto=True,
    )

    coverage, evaluate, target = build_commands(args)

    assert "scripts/check_ruler_coverage.py" in coverage
    assert "--summary-file" in coverage
    assert str(Path(args.output_dir) / "coverage_summary.json") in coverage
    assert "scripts/evaluate_checkpoint.py" in evaluate
    assert "--base-model" in evaluate
    assert "/models/qwen3-235b" in evaluate
    assert "--summary-file" in evaluate
    assert "--details-file" in evaluate
    assert str(Path(args.output_dir) / "test_details.jsonl") in evaluate
    assert "--device-map-auto" in evaluate
    assert str(Path(args.output_dir) / "test_summary.json") in evaluate
    assert "scripts/check_ruler_200k_target.py" in target
    assert "--coverage-summary" in target
    assert "--min-rows" in target
    assert "64" in target
