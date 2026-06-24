"""Archive accepted VLM compactor benchmark results."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_kv.eval.vlm_target import (
    DEFAULT_MIN_COMPACT_VS_FULL_ACCURACY,
    check_vlm_target,
)


def _safe_name(value: str) -> str:
    cleaned = (
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in value
    )
    return "".join(cleaned).strip("-")


def _format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{100.0 * float(value):.2f}%"
    return "n/a"


def _resolve_path(candidate: str, *, source_summary: Path) -> Path:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(path)

    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in source_summary.parents)
    candidates.append(source_summary.parent / path.name)
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(path)


def _details_path(summary: dict[str, Any], summary_path: Path, explicit: str = "") -> Path | None:
    candidate = explicit or str(summary.get("details_file") or "")
    if not candidate:
        return None
    return _resolve_path(candidate, source_summary=summary_path)


def archive_vlm_result(
    *,
    summary_path: str | Path,
    details_path: str | Path | None = None,
    archive_dir: str | Path = "reports/vlm_compactor",
    run_name: str = "",
    min_compact_vs_full_accuracy: float = DEFAULT_MIN_COMPACT_VS_FULL_ACCURACY,
    require_group_targets: bool = True,
) -> Path:
    """Archive a VLM result only after compact-vs-full target checks pass."""
    source_summary = Path(summary_path).expanduser().resolve()
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    checks = check_vlm_target(
        summary,
        min_compact_vs_full_accuracy=min_compact_vs_full_accuracy,
        require_group_targets=require_group_targets,
    )
    failed = [(name, detail) for name, passed, detail in checks if not passed]
    if failed:
        detail = "; ".join(f"{name}: {message}" for name, message in failed)
        raise ValueError(f"VLM result does not satisfy archive target: {detail}")

    if not run_name:
        model = _safe_name(str(summary.get("model") or "vlm")) or "vlm"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{model}-{stamp}"
    destination = Path(archive_dir) / _safe_name(run_name)
    destination.mkdir(parents=True, exist_ok=False)

    archived_summary = destination / "summary.json"
    shutil.copy2(source_summary, archived_summary)
    source_details = _details_path(summary, source_summary, str(details_path or ""))
    archived_details = None
    if source_details is not None:
        archived_details = destination / "details.jsonl"
        shutil.copy2(source_details, archived_details)

    manifest = {
        "source_summary": str(source_summary),
        "source_details": str(source_details) if source_details is not None else "",
        "summary": "summary.json",
        "details": "details.jsonl" if archived_details is not None else "",
        "min_compact_vs_full_accuracy": min_compact_vs_full_accuracy,
        "checks": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in checks
        ],
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    agreement = summary.get("compact_full_agreement")
    if agreement is None:
        agreement = summary.get("compact_vs_full_accuracy")
    readme = destination / "README.md"
    readme.write_text(
        "# VLM Compactor Accepted Result\n\n"
        f"- Model: `{summary.get('model', 'unknown')}`\n"
        f"- Checkpoint: `{summary.get('checkpoint', '')}`\n"
        f"- Rows: `{summary.get('count', summary.get('rows', 0))}`\n"
        f"- Full-cache accuracy: `{_format_percent(summary.get('full_accuracy'))}`\n"
        f"- Compact-cache accuracy: `{_format_percent(summary.get('compact_accuracy'))}`\n"
        "- Compact/full agreement: "
        f"`{_format_percent(agreement)}`\n"
        "- Required compact/full ratio: "
        f"`{_format_percent(min_compact_vs_full_accuracy)}`\n"
        f"- Summary: `{archived_summary.name}`\n"
        f"- Details: `{archived_details.name if archived_details else 'not archived'}`\n",
        encoding="utf-8",
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to evaluate_vlm_compactor.py summary JSON")
    parser.add_argument("--details-file", default="")
    parser.add_argument("--archive-dir", default="reports/vlm_compactor")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--min-compact-vs-full-accuracy",
        type=float,
        default=DEFAULT_MIN_COMPACT_VS_FULL_ACCURACY,
    )
    parser.add_argument("--no-group-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        destination = archive_vlm_result(
            summary_path=args.summary,
            details_path=args.details_file or None,
            archive_dir=args.archive_dir,
            run_name=args.run_name,
            min_compact_vs_full_accuracy=args.min_compact_vs_full_accuracy,
            require_group_targets=not args.no_group_gate,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(destination)


if __name__ == "__main__":
    main()
