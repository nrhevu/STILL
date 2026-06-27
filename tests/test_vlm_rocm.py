import os

import pytest

from neural_kv.utils.rocm import ensure_last_four_gpu_visibility


def test_ensure_last_four_gpu_visibility_sets_default(monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    visible = ensure_last_four_gpu_visibility()

    assert visible == "4,5,6,7"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "4,5,6,7"


def test_ensure_last_four_gpu_visibility_accepts_user_selected_gpus(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5,6,7")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    visible = ensure_last_four_gpu_visibility()

    assert visible == "5,6,7"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "5,6,7"


def test_ensure_last_four_gpu_visibility_rejects_malformed_gpu_list(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5,gpu6,7")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(ValueError, match="not a physical GPU index"):
        ensure_last_four_gpu_visibility()
