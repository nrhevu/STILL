"""Project-local Hugging Face cache defaults."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_hf_home() -> Path:
    return project_root() / "data" / "hf_cache"


def _under_home(value: str) -> bool:
    try:
        path = Path(value).expanduser().resolve()
        home = Path.home().resolve()
        return path == home or home in path.parents
    except OSError:
        return False


def _set_cache_env(key: str, value: Path, *, force_home_replacement: bool = True) -> None:
    current = os.environ.get(key)
    if current is None or (force_home_replacement and _under_home(current)):
        os.environ[key] = str(value)


def configure_hf_cache(hf_home: str | Path | None = None) -> Path:
    """Keep Hugging Face caches under the project scratch workspace by default."""
    home = Path(hf_home) if hf_home is not None else default_hf_home()
    home = home.expanduser().resolve()

    if hf_home is not None:
        os.environ["HF_HOME"] = str(home)
        os.environ["HF_HUB_CACHE"] = str(home)
        os.environ["HF_DATASETS_CACHE"] = str(home / "datasets")
        return home

    _set_cache_env("HF_HOME", home)
    _set_cache_env("HF_HUB_CACHE", home)
    _set_cache_env("HF_DATASETS_CACHE", home / "datasets")
    return Path(os.environ["HF_HOME"]).expanduser().resolve()
