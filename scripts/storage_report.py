#!/usr/bin/env python3
"""Print project-controlled storage usage."""

from neural_kv.storage import check_storage_quota, default_storage_roots

if __name__ == "__main__":
    report = check_storage_quota(default_storage_roots(), "10TB")
    print(report.summary())
