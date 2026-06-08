# Neural KV Compressor Performance Report - 2026-06-08

## Summary

Extended training completed on AMD ROCm with Qwen/Qwen3-4B, 8192-token context, 1024 learned latent KV tokens, and no exact-anchor tokens. The run used the corrected cache-isolated training path and continued from the prior pure checkpoint at global step 300.

Best sampled-validation checkpoint: `checkpoints/qwen3_4b_sec_random_visible_6k_cachefix_pure1024_identity_resume1200_b2_lr5e6_w05/step_800.pt`

Recommended checkpoint: `step_800.pt` for validation-selected use. `final.pt` is also strong and slightly better on the 600-row test split, but validation selection favors `step_800.pt`.

## Data And Storage

| Item | Value |
| --- | ---: |
| Train split | 6,000 rows |
| Validation split | 600 rows |
| Test split | 600 rows |
| Train teacher traces | 6,000 rows, 5,973 tail-gold hits |
| Validation teacher traces | 600 rows, 599 tail-gold hits |
| Test teacher traces | 600 rows, 596 tail-gold hits |
| Workspace data size | 12G |
| Workspace checkpoint size | 32G |
| Workspace cache size | 1.6G |
| Quota status | 72.87GB used / 10TB quota |

## Training

Key settings:

| Setting | Value |
| --- | --- |
| Device path | ROCm via `scripts/rocm_docker_run.sh`, Torch device `cuda`/HIP |
| Model | `Qwen/Qwen3-4B` |
| Init checkpoint | 3k corrected pure run `step_300.pt` |
| Train file | `data/sec_facts_random_visible_6k/train_traces_6000_512.jsonl` |
| Context length | 8192 |
| Latents | 1024 |
| Compression | 8.0x |
| Exact anchors | 0 |
| Batch size | 2 |
| Learning rate | 5e-6 |
| Objective | token KL + reverse KL 0.5 + CE 0.1 + auxiliary letter loss 0.05 |
| Continuation steps | 1200 |
| Global steps reached | 1500 |
| Training time | 4811.43 seconds |

## Sampled Validation During Training

Training sampled validation used 128 validation rows with no-thinking letter scoring, matching `train_still.py` defaults because `--eval-enable-thinking` was not set.

| Global step | Compact acc | Full acc | No-context acc | KL | CE | Reverse KL | Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 0.953125 | 1.000000 | 0.335938 | 0.086548 | 0.117078 | 0.205291 | 0.204255 |
| 500 | 0.921875 | 1.000000 | 0.335938 | 0.058867 | 0.095768 | 0.092088 | 0.114489 |
| 600 | 0.976563 | 1.000000 | 0.335938 | 0.066150 | 0.107203 | 0.106256 | 0.130207 |
| 700 | 0.929688 | 1.000000 | 0.335938 | 0.069947 | 0.117658 | 0.180029 | 0.171743 |
| 800 | 0.984375 | 1.000000 | 0.335938 | 0.081762 | 0.105518 | 0.310424 | 0.319094 |
| 900 | 0.976563 | 1.000000 | 0.335938 | 0.050755 | 0.097214 | 0.073245 | 0.099117 |
| 1000 | 0.968750 | 1.000000 | 0.335938 | 0.047853 | 0.079493 | 0.067183 | 0.089405 |
| 1100 | 0.968750 | 1.000000 | 0.335938 | 0.041995 | 0.066146 | 0.108552 | 0.102886 |
| 1200 | 0.953125 | 1.000000 | 0.335938 | 0.027820 | 0.058106 | 0.076118 | 0.071733 |
| 1300 | 0.960938 | 1.000000 | 0.335938 | 0.037835 | 0.061608 | 0.159722 | 0.123862 |
| 1400 | 0.960938 | 1.000000 | 0.335938 | 0.037172 | 0.073038 | 0.090111 | 0.089558 |
| 1500 | 0.960938 | 1.000000 | 0.335938 | 0.038485 | 0.064262 | 0.083023 | 0.086455 |

## Full MCQ Evaluation

Full MCQ evaluation used no-thinking letter scoring to match the sampled training evaluation.

| Checkpoint | Split | Rows | Compact acc | Full acc | No-context acc | Compression |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `step_800.pt` | Validation | 600 | 0.966667 | 0.996667 | 0.318333 | 8.0 |
| `step_800.pt` | Test | 600 | 0.958333 | 1.000000 | 0.330000 | 8.0 |
| `final.pt` | Validation | 600 | 0.965000 | 0.996667 | 0.318333 | 8.0 |
| `final.pt` | Test | 600 | 0.965000 | 1.000000 | 0.330000 | 8.0 |

## CE Utilization

CE utilization used thinking-enabled teacher-response traces, matching the training target formatting. This was run on a fixed 200-row prefix sample per split.

| Checkpoint | Split | Rows | Target tokens | Full CE | Compact CE | No-context CE | Compact KL | CE utilization | Compression |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `step_800.pt` | Validation traces | 200 | 47,314 | 0.024683 | 0.094268 | 1.162619 | 0.064519 | 0.938850 | 8.0 |
| `step_800.pt` | Test traces | 200 | 47,352 | 0.025552 | 0.106161 | 1.191687 | 0.077178 | 0.930875 | 8.0 |

## Conclusion

The larger-data, longer-step ROCm run exceeded the article-style MCQ and CE-utilization targets on this SEC MCQ benchmark:

| Target | Result |
| --- | --- |
| 8x compression | Met: 8.0x |
| MCQ accuracy around 0.85 | Met: 0.9667 validation / 0.9583 test for `step_800.pt` |
| CE utilization around 0.93 | Met: 0.9389 validation sample / 0.9309 test sample |

Use `step_800.pt` as the validation-selected checkpoint. Keep `final.pt` as a backup because it is nearly tied on validation and slightly better on the 600-row test split.
