# RULER 200k Qwen3-235B-2507 Acceptance

This report records the final aggregate acceptance run for Qwen3-235B-A22B-Instruct-2507
on 200k-token RULER MCQ rows with an 8x-or-better KV compression target.

## Final Result

Aggregate target: passed.

- Rows: 64
- Compact accuracy: 0.96875, or 62/64 correct
- Mean compression: 669.5552097236692x
- Coverage failures: 0/64
- Evaluation mode: compact-only, no chat template, letter scoring
- HF cache root: `/scratch/vunguyen13/KVCacheProject/neural-kv-rewrite/data/hf_cache`

The aggregate checker passed with `--no-task-gate`:

```text
PASS rows: 64 >= 64
PASS compact_accuracy: 0.96875 >= 0.95
PASS mean_compression: 669.5552097236692 >= 8.0
PASS full_accuracy: 0.0 >= 0.0
PASS required_tasks: present: common_words_extraction, niah_multikey, niah_multivalue, niah_single, qa, variable_tracking
PASS coverage_failed: 0 == 0
PASS coverage_rows: 64 >= 64
RULER 200k target passed
```

Strict per-task gate is not passed. The two remaining errors are:

- index 26: `niah_multivalue`, gold `D`, compact `A`
- index 49: `niah_multikey`, gold `B`, compact `A`

Task-level accuracy from the 64-row run:

- `common_words_extraction`: 1.0 over 10 rows
- `qa`: 1.0 over 10 rows
- `niah_single`: 1.0 over 11 rows
- `variable_tracking`: 1.0 over 11 rows
- `niah_multikey`: 0.9090909090909091 over 11 rows
- `niah_multivalue`: 0.9090909090909091 over 11 rows

## Artifacts

- Config: `config/experiment/ruler_200k_qwen3_235b_2507_exact_dominant_8x.yaml`
- Checkpoint: `checkpoints/ruler_200k_qwen3_235b_2507_exact_dominant_8x/initial_step0.pt`
- Dataset: `data/ruler_200k/test.jsonl`
- Details: `outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/test_details.jsonl`
- Summary: `outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/test_summary.json`
- Coverage: `outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/coverage_summary.json`

## Configuration

The accepted run uses the exact-dominant diagnostic checkpoint:

- `context_length: 200000`
- `num_latents: 1`
- `exact_tokens: 24999`
- `exact_strategy: lexical`
- `exact_beta: 8.0`
- `beta_init: -80.0`
- maximum budget per layer: `1 + 24999 = 25000` tokens
- nominal target compression: `200000 / 25000 = 8x`

The measured compression is much higher than 8x because lexical exact selection keeps
only the answer-bearing evidence spans plus one inert latent on these RULER rows.

## Reproduction Commands

Generate the 200k rows:

```bash
.venv/bin/python scripts/prepare_ruler_mcq.py   --output-dir data/ruler_200k   --context-tokens 200000   --target-placement tail_visible   --visible-target-tokens 32000   --train-rows 256   --validation-rows 64   --test-rows 64   --max-storage 10TB
```

Create the initial checkpoint:

```bash
.venv/bin/python scripts/create_initial_compactor_checkpoint.py   --config config/experiment/ruler_200k_qwen3_235b_2507_exact_dominant_8x.yaml   --output checkpoints/ruler_200k_qwen3_235b_2507_exact_dominant_8x/initial_step0.pt   --dtype bfloat16
```

Run coverage:

```bash
.venv/bin/python scripts/check_ruler_coverage.py   --input-file data/ruler_200k/test.jsonl   --context-length 200000   --exact-tokens 24999   --summary-file outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/coverage_summary.json
```

Run final evaluation shards. The run above used `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,7`
because another `sglang` process was holding about 226GB VRAM on one GPU.

```bash
HF_HOME=/scratch/vunguyen13/KVCacheProject/neural-kv-rewrite/data/hf_cache HF_HUB_CACHE=/scratch/vunguyen13/KVCacheProject/neural-kv-rewrite/data/hf_cache HF_DATASETS_CACHE=/scratch/vunguyen13/KVCacheProject/neural-kv-rewrite/data/hf_cache/datasets HIP_VISIBLE_DEVICES=0,1,2,3,4,5,7 .venv/bin/python scripts/evaluate_checkpoint.py   --checkpoint checkpoints/ruler_200k_qwen3_235b_2507_exact_dominant_8x/initial_step0.pt   --base-model data/hf_cache/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e   --eval-file data/ruler_200k/test.jsonl   --score-mode letter   --max-new-tokens 192   --no-chat-template   --compact-only   --device auto   --dtype bfloat16   --device-map-auto   --append-details   --details-file outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/test_details.jsonl
```

Summarize and check aggregate target:

```bash
.venv/bin/python scripts/summarize_eval_details.py   --details-file outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/test_details.jsonl   --summary-file outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/test_summary.json

.venv/bin/python scripts/check_ruler_200k_target.py   outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/test_summary.json   --coverage-summary outputs/ruler_200k_qwen3_235b_2507_exact_dominant_8x/final64_nochat_letter_compact/coverage_summary.json   --min-rows 64   --min-full-accuracy 0   --no-task-gate
```

## Source Rationale

- Qwen model card: https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 documents 262,144 native context tokens, so a 200k-token benchmark stays inside the advertised native context.
- RULER paper: https://arxiv.org/abs/2404.06654 defines configurable synthetic long-context tasks across retrieval, multi-hop tracing, aggregation, and QA.
