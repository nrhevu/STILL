from scripts.summarize_eval_details import summarize_details


def test_summarize_eval_details_matches_checkpoint_metrics_shape() -> None:
    rows = [
        {
            "index": 0,
            "task": "niah_single",
            "gold": "A",
            "no_context": "B",
            "full": "A",
            "compact": "A",
            "compression": 8.0,
        },
        {
            "index": 1,
            "task": "qa",
            "gold": "C",
            "no_context": "C",
            "full": "D",
            "compact": "B",
            "compression": 10.0,
        },
    ]

    summary = summarize_details(rows)

    assert summary["rows"] == 2
    assert summary["compact_accuracy"] == 0.5
    assert summary["full_accuracy"] == 0.5
    assert summary["no_context_accuracy"] == 0.5
    assert summary["mean_compression"] == 9.0
    assert summary["compact_prediction_counts"] == {"A": 1, "B": 1}
    assert summary["task_accuracy"] == {"niah_single": 1.0, "qa": 0.0}
    assert summary["task_counts"] == {"niah_single": 1, "qa": 1}
