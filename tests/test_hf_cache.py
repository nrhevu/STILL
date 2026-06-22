import os
from pathlib import Path

from neural_kv.utils.hf_cache import configure_hf_cache, default_hf_home, project_root


def test_default_hf_home_is_project_local() -> None:
    assert default_hf_home() == project_root() / "data" / "hf_cache"
    assert default_hf_home().is_absolute()


def test_configure_hf_cache_sets_all_hf_cache_envs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    home = configure_hf_cache(tmp_path / "hf_cache")

    assert os.environ["HF_HOME"] == str(home)
    assert os.environ["HF_HUB_CACHE"] == str(home)
    assert os.environ["HF_DATASETS_CACHE"] == str(home / "datasets")


def test_configure_hf_cache_replaces_home_cache(monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    monkeypatch.setenv("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    monkeypatch.setenv(
        "HF_DATASETS_CACHE",
        str(Path.home() / ".cache" / "huggingface" / "datasets"),
    )

    home = configure_hf_cache()

    assert home == default_hf_home()
    assert os.environ["HF_HOME"] == str(default_hf_home())
    assert os.environ["HF_HUB_CACHE"] == str(default_hf_home())
    assert os.environ["HF_DATASETS_CACHE"] == str(default_hf_home() / "datasets")
