#!/usr/bin/env python3
"""Run VLM full-cache or compact-cache evaluation."""

from neural_kv.utils.rocm import ensure_last_four_gpu_visibility


if __name__ == "__main__":
    visible = ensure_last_four_gpu_visibility()
    print(f"HIP_VISIBLE_DEVICES={visible}")
    from neural_kv.eval.vlm import main

    main()
