# Rewrite Legacy Parity Report - 2026-06-21

This report verifies that the clean `neural-kv` rewrite evaluates the legacy selected checkpoint with the same reported SEC 6k metrics as the original codebase.

## Environment

- Worktree: `/scratch/vunguyen13/KVCacheProject/neural-kv-rewrite`
- Branch: `rewrite/neural-kv-clean`
- Checkpoint: `/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/checkpoints/qwen3_4b_sec_random_visible_6k_cachefix_pure1024_identity_resume1200_b2_lr5e6_w05/step_800.pt`
- Data: `/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/data/sec_facts_random_visible_6k`
- Base model cache: `HF_HOME=/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/data/hf_cache`
- Device: `HIP_VISIBLE_DEVICES=7`, after `rocm-smi --showuse` showed GPU 7 at `0%` utilization before each run.

## MCQ Evaluation

Command shape:

```bash
HIP_VISIBLE_DEVICES=7 \
HF_HOME=/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/data/hf_cache \
/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/.uv-bootstrap/bin/uv run --extra train --extra dev \
python scripts/evaluate.py \
  --checkpoint /scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/checkpoints/qwen3_4b_sec_random_visible_6k_cachefix_pure1024_identity_resume1200_b2_lr5e6_w05/step_800.pt \
  --eval-file <validation-or-test.jsonl> \
  --limit 600 \
  --score-mode letter \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

| Split | Rows | Compact acc | Full acc | No-context acc | Compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 600 | 0.9666666666666667 | 0.9966666666666667 | 0.31833333333333336 | 8.0 |
| Test | 600 | 0.9583333333333334 | 1.0 | 0.33 | 8.0 |

These match the legacy report values: validation `0.966667 / 0.996667 / 0.318333 / 8.0`, test `0.958333 / 1.000000 / 0.330000 / 8.0`.

## CE Utilization

Command shape:

```bash
HIP_VISIBLE_DEVICES=7 \
HF_HOME=/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/data/hf_cache \
/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/.uv-bootstrap/bin/uv run --extra train --extra dev \
python scripts/evaluate_ce_utilization.py \
  --checkpoint /scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/checkpoints/qwen3_4b_sec_random_visible_6k_cachefix_pure1024_identity_resume1200_b2_lr5e6_w05/step_800.pt \
  --eval-file <validation-or-test-traces.jsonl> \
  --limit 200 \
  --enable-thinking \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

| Split | Rows | Target tokens | Full CE | Compact CE | No-context CE | Compact KL | CE utilization | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation traces | 200 | 47314.0 | 0.024683279471296563 | 0.09426800201534746 | 1.16261893047224 | 0.06451916353622021 | 0.9388500373612135 | 8.0 |
| Test traces | 200 | 47352.0 | 0.025551985274682654 | 0.10616106582125157 | 1.1916868882551148 | 0.07717799584521791 | 0.9308749953881438 | 8.0 |

These match the legacy report values: validation CE utilization `0.938850`, test CE utilization `0.930875`, with matching CE/KL/compression values to reported precision.
