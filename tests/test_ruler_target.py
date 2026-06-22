from neural_kv.eval.ruler_target import DEFAULT_REQUIRED_TASKS, check_ruler_target


def _passing_payload() -> dict[str, object]:
    return {
        "rows": 64,
        "compact_accuracy": 0.95,
        "full_accuracy": 0.99,
        "mean_compression": 8.0,
        "task_accuracy": {task: 0.95 for task in DEFAULT_REQUIRED_TASKS},
        "task_counts": {task: 1 for task in DEFAULT_REQUIRED_TASKS},
    }


def test_ruler_target_passes_required_accuracy_and_compression() -> None:
    checks = check_ruler_target(_passing_payload())

    assert all(passed for _, passed, _ in checks)


def test_ruler_target_fails_below_95_compact_accuracy() -> None:
    payload = _passing_payload()
    payload["compact_accuracy"] = 0.949

    failed = {name for name, passed, _ in check_ruler_target(payload) if not passed}
    assert "compact_accuracy" in failed


def test_ruler_target_fails_hidden_per_task_drop() -> None:
    payload = _passing_payload()
    payload["task_accuracy"] = {
        task: (0.90 if task == "qa" else 1.0)
        for task in DEFAULT_REQUIRED_TASKS
    }

    failed = {name for name, passed, _ in check_ruler_target(payload) if not passed}
    assert "task_accuracy" in failed


def test_ruler_target_fails_missing_required_task() -> None:
    payload = _passing_payload()
    payload["task_counts"] = {"niah_single": 64}

    failed = {name for name, passed, _ in check_ruler_target(payload) if not passed}
    assert "required_tasks" in failed


def test_ruler_target_fails_tiny_eval_summary() -> None:
    payload = _passing_payload()
    payload["rows"] = 8

    failed = {name for name, passed, _ in check_ruler_target(payload) if not passed}
    assert "rows" in failed


def test_ruler_target_fails_failed_coverage_summary() -> None:
    payload = _passing_payload()
    coverage = {"rows": 64, "failed": 1}

    failed = {
        name
        for name, passed, _ in check_ruler_target(payload, coverage_payload=coverage)
        if not passed
    }
    assert "coverage_failed" in failed
