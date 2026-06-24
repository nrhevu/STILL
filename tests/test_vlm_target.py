from neural_kv.eval.vlm_target import check_vlm_target


def test_check_vlm_target_accepts_95_percent_full_cache_agreement() -> None:
    payload = {
        "count": 100,
        "full_reference_count": 100,
        "full_accuracy": 0.20,
        "compact_accuracy": 0.19,
        "compact_full_agreement": 0.95,
        "groups": [
            {
                "task": "mmmu",
                "resolution": 448,
                "image_token_budget": 256,
                "full_reference_count": 100,
                "full_accuracy": 0.20,
                "compact_accuracy": 0.19,
                "compact_full_agreement": 0.95,
            }
        ],
    }

    checks = check_vlm_target(payload, min_compact_vs_full_accuracy=0.95)

    assert all(passed for _, passed, _ in checks)


def test_check_vlm_target_rejects_below_95_percent_group_agreement() -> None:
    payload = {
        "count": 100,
        "full_reference_count": 100,
        "full_accuracy": 0.80,
        "compact_accuracy": 0.76,
        "compact_full_agreement": 0.96,
        "groups": [
            {
                "task": "docvqa",
                "resolution": 448,
                "image_token_budget": 256,
                "full_reference_count": 10,
                "full_accuracy": 0.80,
                "compact_accuracy": 0.72,
                "compact_full_agreement": 0.90,
            }
        ],
    }

    checks = check_vlm_target(payload, min_compact_vs_full_accuracy=0.95)

    assert not dict((name, passed) for name, passed, _ in checks)[
        "group_compact_full_agreement"
    ]


def test_check_vlm_target_requires_full_reference_rows() -> None:
    payload = {
        "count": 1,
        "full_reference_count": 0,
        "full_accuracy": 0.0,
        "compact_accuracy": 0.0,
        "compact_full_agreement": None,
        "groups": [],
    }

    checks = dict((name, passed) for name, passed, _ in check_vlm_target(payload))

    assert checks["full_reference_rows"] is False
    assert checks["compact_full_agreement"] is False
