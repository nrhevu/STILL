import json

from neural_kv.eval.vlm_archive import archive_vlm_result


def test_archive_vlm_result_copies_only_passing_summary(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    details = tmp_path / "details.jsonl"
    details.write_text("{}\n", encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "count": 10,
                "model": "unit/model",
                "checkpoint": "checkpoint.ckpt",
                "details_file": str(details),
                "full_accuracy": 0.8,
                "compact_accuracy": 0.76,
                "compact_full_agreement": 0.95,
                "groups": [
                    {
                        "task": "mmmu",
                        "resolution": 448,
                        "image_token_budget": 256,
                        "full_accuracy": 0.8,
                        "compact_accuracy": 0.76,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    archived = archive_vlm_result(
        summary_path=summary,
        archive_dir=tmp_path / "reports",
        run_name="accepted",
    )

    assert (archived / "summary.json").exists()
    assert (archived / "details.jsonl").exists()
    assert (archived / "archive_manifest.json").exists()
    assert "Compact/full agreement" in (archived / "README.md").read_text()


def test_archive_vlm_result_rejects_below_target(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "count": 10,
                "full_accuracy": 0.8,
                "compact_accuracy": 0.7,
                "groups": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        archive_vlm_result(summary_path=summary, archive_dir=tmp_path / "reports")
    except ValueError as exc:
        assert "does not satisfy" in str(exc)
    else:
        raise AssertionError("below-target result should not be archived")



def test_archive_vlm_result_resolves_evaluator_relative_details(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs" / "qwen3_vl_compactor_bench"
    output_dir.mkdir(parents=True)
    summary = output_dir / "summary.json"
    details = output_dir / "details.jsonl"
    details.write_text("{}\n", encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "count": 1,
                "model": "unit/model",
                "details_file": "outputs/qwen3_vl_compactor_bench/details.jsonl",
                "full_accuracy": 1.0,
                "compact_accuracy": 0.95,
                "compact_full_agreement": 0.95,
                "groups": [
                    {
                        "task": "mmmu",
                        "resolution": 448,
                        "image_token_budget": 256,
                        "full_accuracy": 1.0,
                        "compact_accuracy": 0.95,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    archived = archive_vlm_result(
        summary_path=summary,
        archive_dir=tmp_path / "reports",
        run_name="relative-details",
    )

    assert (archived / "details.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_archive_vlm_result_requires_referenced_details(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "count": 1,
                "details_file": "missing-details.jsonl",
                "full_accuracy": 1.0,
                "compact_accuracy": 0.95,
                "compact_full_agreement": 0.95,
                "groups": [
                    {
                        "task": "mmmu",
                        "resolution": 448,
                        "image_token_budget": 256,
                        "full_accuracy": 1.0,
                        "compact_accuracy": 0.95,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        archive_vlm_result(summary_path=summary, archive_dir=tmp_path / "reports")
    except FileNotFoundError as exc:
        assert "missing-details" in str(exc)
    else:
        raise AssertionError("referenced details file should be required")
