import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_performance_target.py"


def run_target(
    tmp_path: Path, payload: dict[str, float], *args: str
) -> subprocess.CompletedProcess[str]:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(summary), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_target_gate_accepts_niah_summary_schema(tmp_path: Path) -> None:
    result = run_target(
        tmp_path,
        {
            "overall_success_rate": 1.0,
            "overall_full_success_rate": 1.0,
            "overall_mean_compression": 875.46,
            "overall_relative_success_to_full": 1.0,
        },
        "--min-relative-accuracy",
        "0.95",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS relative_accuracy_to_full: 1.0 >= 0.95" in result.stdout
    assert "SKIP mcq_utilization: not present for NIAH summary schema" in result.stdout


def test_target_gate_keeps_legacy_summary_schema(tmp_path: Path) -> None:
    result = run_target(
        tmp_path,
        {
            "compact_accuracy": 0.95,
            "full_accuracy": 1.0,
            "no_context_accuracy": 0.25,
            "mean_compression": 12.0,
        },
        "--min-relative-accuracy",
        "0.90",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS compact_accuracy: 0.95 >= 0.85" in result.stdout
    assert "PASS mcq_utilization:" in result.stdout
