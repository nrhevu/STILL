# Qwen3-32B x8 NIAH Compactor Report - 2026-06-09

## Summary

Trained a NeuralKVCompressor compactor for `Qwen/Qwen3-32B` on synthetic
needle-in-a-haystack retrieval while keeping the KV cache at about 8x
compression. The best checkpoint is the low-learning-rate continuation from the
first high-LR checkpoint.

| Metric | Value |
| --- | ---: |
| Best checkpoint | `checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25/final.pt` |
| Base model | `Qwen/Qwen3-32B` |
| Cache source tokens | ~8,041-8,042 |
| Compact cache tokens | 1,005 per layer |
| Mean compression | 8.001x validation / 8.001x test |
| Validation MCQ compact accuracy | 1.000 |
| Held-out test MCQ compact accuracy | 1.000 |
| Free-form compact NIAH exact rate | 1.000 |
| Storage after runs | 227.50GB used / 10TB quota |

## Data

Generated NIAH MCQ data under:

```text
data/niah_qwen3_32b_ctx8192_chat/
```

Split sizes:

| Split | Rows |
| --- | ---: |
| Train | 600 |
| Validation | 100 |
| Test | 100 |

The data uses depths `0,25,50,75,100`, a cache context budget of 8192 tokens,
and a 192-token raw-context margin so the chat-template context keeps the
needle visible even at depth 100.

## Compactor Configuration

| Setting | Value |
| --- | --- |
| `num_latents` | 941 |
| `exact_tokens` | 64 |
| `exact_strategy` | `lexical` |
| Total compact tokens | 1005 |
| `context_length` | 8192 |
| `beta_base` | `zero` |
| Loss | letter KL + reverse KL 0.5 + CE 0.1 |
| Device | ROCm via `scripts/rocm_docker_run.sh`, Torch device `cuda` |
| Dtype | `bfloat16` |

This preserves the query-matched needle line as exact KV tokens and uses learned
latents for the rest of the cache. The lexical exact selector was fixed to fall
back through decoded context text when token-subsequence matching fails at the
end of chat-formatted contexts; this fixed depth-100 free-form retrieval.

## Training Runs

| Run | Init | LR | Result |
| --- | --- | ---: | --- |
| `qwen3_32b_niah_ctx8192_x8_lex64_lat941` | scratch | 5e-5 | step 25 reached 0.94 validation, later steps degraded |
| `qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25` | `step_25.pt` | 5e-6 | final/global step 100 reached 1.00 validation |
| `qwen3_32b_niah_ctx8192_x8_lex128_lat877` | scratch | 5e-6 | lower validation; not selected |

Final selected validation metrics:

| Compact acc | Full acc | No-context acc | Mean compression | Train seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1.000 | 1.000 | 0.280 | 8.001 | 802.66 |

Held-out MCQ test metrics:

| Compact acc | Full acc | No-context acc | Mean compression |
| ---: | ---: | ---: | ---: |
| 1.000 | 1.000 | 0.220 | 8.001 |

## Free-Form Compact NIAH Generation

Free-form generation used the selected compactor checkpoint with compact-cache
generation, not full-cache generation. It evaluated 15 cases: five depths and
three trials each.

| Depth % | Trials | Success rate | Exact rate | Cache tokens | Compression | Mean decode time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 1.000 | 1.000 | 1005 | 8.001 | 0.62s |
| 25 | 3 | 1.000 | 1.000 | 1005 | 8.002 | 0.64s |
| 50 | 3 | 1.000 | 1.000 | 1005 | 8.002 | 0.60s |
| 75 | 3 | 1.000 | 1.000 | 1005 | 8.002 | 0.60s |
| 100 | 3 | 1.000 | 1.000 | 1005 | 8.001 | 0.61s |

Overall free-form compact result: `1.000`
success and `1.000` exact match at
`8.001x` mean compression.

## Reproduction Commands

```bash
HF_HOME=data/hf_cache .uv-bootstrap/bin/uv run python scripts/prepare_niah_mcq.py \
  --model Qwen/Qwen3-32B \
  --output-dir data/niah_qwen3_32b_ctx8192_chat \
  --context-length 8192 \
  --raw-context-token-margin 192 \
  --train-rows 600 \
  --validation-rows 100 \
  --test-rows 100 \
  --depths 0,25,50,75,100 \
  --max-storage 10TB

HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/train_still.py \
  --model Qwen/Qwen3-32B \
  --train-file data/niah_qwen3_32b_ctx8192_chat/train.jsonl \
  --eval-file data/niah_qwen3_32b_ctx8192_chat/validation.jsonl \
  --output-dir checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25 \
  --init-checkpoint checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941/step_25.pt \
  --num-latents 941 \
  --exact-tokens 64 \
  --exact-strategy lexical \
  --context-length 8192 \
  --steps 75 \
  --batch-size 1 \
  --learning-rate 0.000005 \
  --kl-weight 1.0 \
  --reverse-kl-weight 0.5 \
  --ce-weight 0.1 \
  --eval-every 25 \
  --save-every 25 \
  --eval-limit 100 \
  --loss-mode letter \
  --target-mode letter \
  --score-mode letter \
  --beta-base zero \
  --balanced-answer-sampling \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB

HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25/final.pt \
  --eval-file data/niah_qwen3_32b_ctx8192_chat/test.jsonl \
  --limit 100 \
  --score-mode letter \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB

HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/evaluate_niah.py \
  --model Qwen/Qwen3-32B \
  --checkpoint checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25/final.pt \
  --context-lengths 8192 \
  --case-context-token-margin 192 \
  --depths 0,25,50,75,100 \
  --trials 3 \
  --output reports/niah_qwen3_32b_x8_compactor_generation_2026-06-09_fixedlex.jsonl \
  --summary-output reports/niah_qwen3_32b_x8_compactor_generation_2026-06-09_fixedlex_summary.json \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 24 \
  --max-storage 10TB
```

## Artifacts

- Best checkpoint: `checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25/final.pt`
- Training summary: `checkpoints/qwen3_32b_niah_ctx8192_x8_lex64_lat941_lr5e6_from25/summary.json`
- Free-form records: `reports/niah_qwen3_32b_x8_compactor_generation_2026-06-09_fixedlex.jsonl`
- Free-form summary: `reports/niah_qwen3_32b_x8_compactor_generation_2026-06-09_fixedlex_summary.json`
- Held-out test summary: `reports/niah_qwen3_32b_x8_compactor_test_2026-06-09_summary.json`
