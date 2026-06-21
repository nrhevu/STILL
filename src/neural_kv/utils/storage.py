"""Storage accounting helpers for bounded data/model runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?i?b?)?\s*$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "b": 1,
    "k": 10**3,
    "kb": 10**3,
    "m": 10**6,
    "mb": 10**6,
    "g": 10**9,
    "gb": 10**9,
    "t": 10**12,
    "tb": 10**12,
    "p": 10**15,
    "pb": 10**15,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
    "pib": 2**50,
}


def parse_size(value: str | int | float) -> int:
    """Parse strings such as ``10TB`` or ``512GiB`` into bytes."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"Invalid size string: {value!r}")
    magnitude = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit not in _UNITS:
        raise ValueError(f"Unsupported size unit in {value!r}")
    return int(magnitude * _UNITS[unit])


def format_bytes(num_bytes: int) -> str:
    """Return a compact decimal byte string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.2f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1000
    return f"{num_bytes}B"


def directory_size(path: str | Path) -> int:
    """Compute a directory's apparent size by summing file sizes."""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def default_storage_roots() -> list[Path]:
    """Return paths that this project commonly uses for large local artifacts."""
    roots = [
        Path("data"),
        Path("checkpoints"),
        Path("artifacts"),
        Path(".venv"),
        Path(".uv-bootstrap"),
        Path(".cache"),
        Path("data/cache"),
        Path("data/comgr_cache"),
        Path("data/torchinductor_cache"),
        Path("data/uv_cache"),
    ]
    uv_cache = os.environ.get("UV_CACHE_DIR")
    roots.append(Path(uv_cache) if uv_cache else Path.home() / ".cache" / "uv")
    return roots


@dataclass(frozen=True)
class StorageReport:
    """Storage usage report for project-controlled paths."""

    used_bytes: int
    quota_bytes: int
    roots: tuple[Path, ...]

    @property
    def remaining_bytes(self) -> int:
        return self.quota_bytes - self.used_bytes

    def summary(self) -> str:
        return (
            f"{format_bytes(self.used_bytes)} used / {format_bytes(self.quota_bytes)} quota "
            f"({format_bytes(max(self.remaining_bytes, 0))} remaining)"
        )


def check_storage_quota(roots: list[str | Path], max_storage: str | int = "10TB") -> StorageReport:
    """Raise if the combined size of ``roots`` exceeds ``max_storage``."""
    quota = parse_size(max_storage)
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        path = Path(root)
        try:
            key = path.resolve()
        except FileNotFoundError:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(path)
    root_paths = tuple(unique_roots)
    used = sum(directory_size(root) for root in root_paths)
    report = StorageReport(used_bytes=used, quota_bytes=quota, roots=root_paths)
    if used > quota:
        raise RuntimeError(
            f"Storage quota exceeded: {report.summary()} across "
            + ", ".join(str(path) for path in root_paths)
        )
    return report
