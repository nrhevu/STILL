from types import SimpleNamespace

import torch
from PIL import Image

from neural_kv.data.vlm import VLMExample
from neural_kv.eval.vlm_compactor import (
    encode_vlm_example,
    image_token_budget_to_pixels,
    summarize_vlm_results,
)


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, *, return_tensors=None, add_special_tokens=False):
        del add_special_tokens
        values = [max(1, ord(char) % 50) for char in text]
        tensor = torch.tensor([values], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor}
        return SimpleNamespace(input_ids=values)

    def decode(self, ids, *, skip_special_tokens=True):
        del skip_special_tokens
        if any(int(item) == 5 for item in ids):
            return "A"
        return ""


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        return "<image>\n" + messages[0]["content"][1]["text"]

    def __call__(self, *, text, images, return_tensors, padding):
        del text, images, return_tensors, padding
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


def test_encode_vlm_example_tracks_visual_tokens_and_sizes() -> None:
    processor = FakeProcessor()
    example = VLMExample(
        id="ex",
        dataset="unit",
        task="mmmu",
        question="Which option?",
        image=Image.new("RGB", (80, 40), "white"),
        choices=["A choice", "B choice", "C choice", "D choice"],
        answer_index=0,
        answers=["A choice"],
    )

    encoded = encode_vlm_example(
        processor,
        example,
        device=torch.device("cpu"),
        resolution=40,
    )

    assert encoded.source_tokens == 4
    assert encoded.visual_tokens == 4
    assert encoded.original_size == (80, 40)
    assert encoded.resized_size == (40, 20)
    assert encoded.prompt_ids.shape[0] == 1


def test_image_token_budget_to_pixels_uses_qwen_patch_area() -> None:
    assert image_token_budget_to_pixels(256) == 256 * 28 * 28


def test_summarize_vlm_results_groups_by_sweep_setting() -> None:
    rows = [
        {
            "task": "mmmu",
            "resolution": 448,
            "image_token_budget": 256,
            "full_correct": True,
            "compact_correct": False,
            "full_reference_valid": True,
            "compact_matches_full": True,
            "source_tokens": 100,
            "visual_tokens": 64,
            "compact_tokens": 25,
            "compression": 4.0,
            "full_seconds": 1.0,
            "compact_seconds": 0.5,
        },
        {
            "task": "mmmu",
            "resolution": 448,
            "image_token_budget": 256,
            "full_correct": True,
            "compact_correct": True,
            "full_reference_valid": True,
            "compact_matches_full": False,
            "source_tokens": 120,
            "visual_tokens": 80,
            "compact_tokens": 30,
            "compression": 4.0,
            "full_seconds": 1.5,
            "compact_seconds": 0.7,
        },
    ]

    summary = summarize_vlm_results(rows)

    assert summary["full_accuracy"] == 1.0
    assert summary["compact_accuracy"] == 0.5
    assert summary["compact_vs_full_accuracy"] == 0.5
    assert summary["compact_full_agreement"] == 0.5
    assert summary["target_passed"] is False
    assert summary["groups"][0]["avg_source_tokens"] == 110
