#!/usr/bin/env python3
"""Print project-controlled storage usage."""

from pathlib import Path

from neural_kv.storage import check_storage_quota

if __name__ == "__main__":
    report = check_storage_quota(
        [Path("data"), Path("checkpoints"), Path("artifacts"), Path(".venv")],
        "10TB",
    )
    print(report.summary())
