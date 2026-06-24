import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_vlm_compactor_acceptance import build_commands, preflight_checks


def test_vlm_acceptance_runner_builds_eval_gate_archive_commands() -> None:
    args = argparse.Namespace(
        config="config/experiment/qwen3_vl_compactor_bench.yaml",
        checkpoint="checkpoints/qwen3-vl/step_100.ckpt",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        output_dir="outputs/vlm_acceptance",
        archive_dir="reports/vlm_compactor",
        run_name="qwen3-vl-accepted",
        limit=3,
        min_rows=3,
        resolutions="448",
        image_token_budgets="256",
        max_new_tokens=16,
        target_compact_vs_full=0.95,
        device="auto",
        dtype="bfloat16",
        device_map_auto=True,
        no_wait_for_gpu=False,
        no_group_gate=False,
        init_random_compactor=False,
        preflight_only=False,
        skip_preflight=False,
        dry_run=False,
    )

    evaluate, check, archive = build_commands(args)

    assert "scripts/evaluate_vlm_compactor.py" in evaluate
    assert "--fail-under-target" in evaluate
    assert "--checkpoint" in evaluate
    assert "checkpoints/qwen3-vl/step_100.ckpt" in evaluate
    assert "--summary-file" in evaluate
    assert str(Path(args.output_dir) / "summary.json") in evaluate
    assert "--details-file" in evaluate
    assert str(Path(args.output_dir) / "details.jsonl") in evaluate
    assert "--target-compact-vs-full" in evaluate
    assert "0.95" in evaluate
    assert "--device-map-auto" in evaluate

    assert "scripts/check_vlm_compactor_target.py" in check
    assert str(Path(args.output_dir) / "summary.json") in check
    assert "--min-compact-vs-full-accuracy" in check
    assert "--min-rows" in check
    assert "3" in check

    assert "scripts/archive_vlm_compactor_result.py" in archive
    assert str(Path(args.output_dir) / "summary.json") in archive
    assert str(Path(args.output_dir) / "details.jsonl") in archive
    assert "--run-name" in archive
    assert "qwen3-vl-accepted" in archive



def test_vlm_acceptance_preflight_reports_missing_checkpoint(monkeypatch) -> None:
    args = argparse.Namespace(
        config="config/experiment/qwen3_vl_compactor_bench.yaml",
        checkpoint="/tmp/definitely_missing_vlm_checkpoint.ckpt",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        output_dir="outputs/vlm_acceptance",
        archive_dir="reports/vlm_compactor",
        run_name="qwen3-vl-accepted",
        limit=1,
        min_rows=1,
        resolutions="448",
        image_token_budgets="256",
        max_new_tokens=16,
        target_compact_vs_full=0.95,
        device="cpu",
        dtype="bfloat16",
        device_map_auto=False,
        no_wait_for_gpu=False,
        no_group_gate=False,
        init_random_compactor=False,
        preflight_only=True,
        skip_preflight=False,
        dry_run=False,
    )
    monkeypatch.setattr(
        "scripts.run_vlm_compactor_acceptance._package_available",
        lambda name: True,
    )

    checks = dict((name, passed) for name, passed, _ in preflight_checks(args))

    assert checks["checkpoint"] is False
    assert checks["package:torch"] is True



def test_vlm_acceptance_runner_supports_init_random_compactor() -> None:
    args = argparse.Namespace(
        config="config/experiment/qwen3_vl_exact_cache_acceptance.yaml",
        checkpoint="",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        output_dir="outputs/vlm_acceptance",
        archive_dir="reports/vlm_compactor",
        run_name="qwen3-vl-exact-cache",
        limit=1,
        min_rows=1,
        resolutions="448",
        image_token_budgets="256",
        max_new_tokens=1,
        target_compact_vs_full=0.95,
        device="auto",
        dtype="bfloat16",
        device_map_auto=True,
        no_wait_for_gpu=False,
        no_group_gate=False,
        init_random_compactor=True,
        preflight_only=False,
        skip_preflight=False,
        dry_run=False,
    )

    evaluate, _, _ = build_commands(args)

    assert "--init-random-compactor" in evaluate
    assert "--checkpoint" not in evaluate
